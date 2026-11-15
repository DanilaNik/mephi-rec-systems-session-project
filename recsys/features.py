"""Этап 1. Расширенные признаки пользователей, книг и взаимодействий.

Признаки делятся на два типа:

* статические признаки книг (`ItemContent`): тематические категории (NMF по тегам), эмбеддинги текстового
  профиля (TruncatedSVD от TF-IDF), автор, серия, метаданные Goodreads. Они не зависят от оценок
  пользователей и считаются один раз;
* признаки, зависящие от оценок (`FeatureStore`): считаются только по обучающей части. Для подбора весов на
  val их строят по train, для финальной оценки - по trainval, поэтому утечки из test нет.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import LIKE_THRESHOLD, Dataset, is_shelf_tag, to_csr

# --------------------------------------------------------------------------- тексты, автор, серия
def build_book_profiles(books: pd.DataFrame, book_tags: pd.DataFrame,
                        max_tags: int = 30, max_repeat: int = 5) -> pd.Series:
    """Текстовый профиль книги: original_title + теги (из ДЗ)."""
    title = books["original_title"].fillna(books["title"].str.replace(r"\s*\(.*\)\s*$", "", regex=True))

    tags = book_tags[~book_tags["tag_name"].fillna("").map(is_shelf_tag)].copy()
    tags = tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
    tags = tags.groupby("goodreads_book_id").head(max_tags)               # самые частые теги книги
    peak = tags.groupby("goodreads_book_id")["count"].transform("max")
    repeat = np.ceil(max_repeat * tags["count"] / peak).astype(int)      # популярный тег повторяем чаще
    token = tags["tag_name"].str.lower().str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    tags["text"] = [" ".join([t] * r) for t, r in zip(token, repeat)]
    tag_text = tags.groupby("goodreads_book_id")["text"].agg(" ".join)

    tag_text = books["goodreads_book_id"].map(tag_text).fillna("")
    return (title.fillna("") + " " + tag_text).str.lower()


def first_author(books: pd.DataFrame) -> pd.Series:
    return books["authors"].fillna("").str.split(",").str[0].str.strip()


def series_name(books: pd.DataFrame) -> pd.Series:
    """Название серии из заголовка вида «Title (Series, #1)». У книг без серии - NaN."""
    return books["title"].str.extract(r"\(([^()]*?),?\s*#\d+[^()]*\)")[0]


def _codes(keys: pd.Series) -> np.ndarray:
    """Целочисленные коды ключей; NaN/пустое значение получает уникальный код (не склеивается с другими)."""
    keys = keys.copy()
    empty = keys.isna() | (keys == "")
    keys[empty] = "__none__" + pd.Series(np.arange(len(keys)), index=keys.index).astype(str)[empty]
    return pd.factorize(keys)[0]


# --------------------------------------------------------------------------- статические признаки книг
@dataclass
class ItemContent:
    """Признаки книг, которые не зависят от оценок пользователей."""
    tfidf: sparse.csr_matrix          # строки нормированы, (книги x термины)
    item_emb: np.ndarray              # (книги x d), L2-нормированные SVD-эмбеддинги текстового профиля
    genre: np.ndarray                 # (книги x n_genres), строки суммируются в 1 (темы NMF по тегам)
    genre_names: list[str]            # человекочитаемые названия тем (по топ-тегам)
    author_code: np.ndarray           # код первого автора
    series_code: np.ndarray           # код серии (у книг без серии - уникальный)
    has_series: np.ndarray            # книга входит в серию
    meta: pd.DataFrame                # метаданные Goodreads: log_gr_count, gr_avg_rating, gr_rating_entropy, ...
    profiles: pd.Series = field(repr=False, default=None)

    @property
    def n_genres(self) -> int:
        return self.genre.shape[1]

    def book_entities(self, codes: np.ndarray) -> sparse.csr_matrix:
        """Матрица книга -> автор/серия (one-hot, разреженная)."""
        n = len(codes)
        return sparse.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), codes)), shape=(n, codes.max() + 1))


def _topic_model(books: pd.DataFrame, book_tags: pd.DataFrame, n_topics: int, seed: int,
                 n_tags: int = 1500) -> tuple[np.ndarray, list[str]]:
    """Тематические категории: NMF по матрице «книга x содержательный тег».

    Берём n_tags самых распространённых содержательных (не «полочных») тегов, вес тега в книге - log1p(count), затем нормировка
    строк. Каждая тема получает название из трёх самых характерных тегов.
    """
    tags = book_tags[~book_tags["tag_name"].fillna("").map(is_shelf_tag)]
    top = tags.groupby("tag_name").size().sort_values(ascending=False).head(n_tags).index
    tags = tags[tags["tag_name"].isin(top)]
    item_of = pd.Series(np.arange(len(books)), index=books["goodreads_book_id"])
    rows = item_of.reindex(tags["goodreads_book_id"]).to_numpy()
    ok = ~np.isnan(rows)
    cols_of = pd.Series(np.arange(len(top)), index=top)
    cols = cols_of.loc[tags["tag_name"]].to_numpy()
    mat = sparse.csr_matrix((np.log1p(tags["count"].to_numpy(dtype=np.float32))[ok],
                             (rows[ok].astype(int), cols[ok])), shape=(len(books), len(top)))
    mat = normalize(mat, norm="l2")
    nmf = NMF(n_components=n_topics, init="nndsvd", random_state=seed, max_iter=300)
    w = nmf.fit_transform(mat)                                        # (книги x темы)
    names = []
    for t in range(n_topics):
        best = np.argsort(-nmf.components_[t])[:3]
        names.append("/".join(str(top[j]) for j in best))
    row_sum = w.sum(axis=1, keepdims=True)
    genre = np.divide(w, row_sum, out=np.full_like(w, 1.0 / n_topics), where=row_sum > 0)
    return genre.astype(np.float32), names


def build_item_content(ds: Dataset, n_topics: int = 20, emb_dim: int = 64, seed: int = 42) -> ItemContent:
    books = ds.books
    profiles = build_book_profiles(books, ds.book_tags)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    tfidf = vec.fit_transform(profiles).astype(np.float32).tocsr()
    d = int(min(emb_dim, tfidf.shape[1] - 1, max(2, tfidf.shape[0] - 1)))
    emb = TruncatedSVD(n_components=d, random_state=seed).fit_transform(tfidf)
    emb = normalize(emb).astype(np.float32)

    n_topics = int(min(n_topics, max(2, len(books) // 5)))
    genre, genre_names = _topic_model(books, ds.book_tags, n_topics, seed)

    authors, series = first_author(books), series_name(books)
    cnt = books[[f"ratings_{r}" for r in range(1, 6)]].to_numpy(dtype=np.float64)
    p = cnt / np.maximum(cnt.sum(axis=1, keepdims=True), 1)
    entropy = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(axis=1) / np.log(5)
    year = books["original_publication_year"].fillna(books["original_publication_year"].median()) \
        if "original_publication_year" in books else pd.Series(0.0, index=books.index)
    meta = pd.DataFrame({
        "log_gr_count": np.log1p(books["ratings_count"].to_numpy(dtype=np.float64)),
        "gr_avg_rating": books["average_rating"].to_numpy(dtype=np.float64),
        "gr_rating_entropy": entropy,                                       # разнообразие оценок книги
        "gr_like_share": (cnt[:, 3] + cnt[:, 4]) / np.maximum(cnt.sum(axis=1), 1),
        "year": year.clip(lower=1800).to_numpy(dtype=np.float64),
    }, index=books.index)
    return ItemContent(tfidf, emb, genre, genre_names, _codes(authors), _codes(series),
                       series.notna().to_numpy(), meta, profiles)


# --------------------------------------------------------------------------- признаки, зависящие от оценок
USER_SEGMENTS = ("new", "regular", "active")


def segment_users(n_ratings: np.ndarray, new_max: int = 4, active_min: int = 30) -> np.ndarray:
    """Тип пользователя по числу оценок в обучающей части: 0 - новый (<5), 1 - обычный, 2 - активный (>=30)."""
    return np.where(n_ratings <= new_max, 0, np.where(n_ratings >= active_min, 2, 1))


@dataclass
class FeatureStore:
    """Признаки пользователей и книг, рассчитанные по одной обучающей выборке."""
    content: ItemContent
    user_features: pd.DataFrame       # индекс u
    item_features: pd.DataFrame       # индекс i
    user_genre: np.ndarray            # (пользователи x темы): профиль тем понравившихся книг, сумма 1 (или 0)
    likes: sparse.csr_matrix          # (пользователи x книги): 1, если оценка >= порога
    user_segment: np.ndarray          # 0/1/2 (см. segment_users)

    @property
    def item_emb(self) -> np.ndarray:
        return self.content.item_emb


def build_features(ds: Dataset, train_df: pd.DataFrame, content: ItemContent,
                   like_threshold: float = LIKE_THRESHOLD, prior_strength: float = 10.0,
                   new_max: int = 4, active_min: int = 30) -> FeatureStore:
    """Расширенные признаки пользователей и книг по обучающей выборке train_df (колонки u, i, rating)."""
    n_users, n_items = ds.shape
    u, i, r = train_df["u"].to_numpy(), train_df["i"].to_numpy(), train_df["rating"].to_numpy(dtype=np.float64)
    like = r >= like_threshold
    gmean = r.mean()
    R_like = to_csr(train_df[like], ds.shape, value=None)

    # ----- признаки пользователей
    n = np.bincount(u, minlength=n_users).astype(np.float64)
    s1 = np.bincount(u, weights=r, minlength=n_users)
    s2 = np.bincount(u, weights=r * r, minlength=n_users)
    n_like = np.bincount(u, weights=like.astype(np.float64), minlength=n_users)
    n_dis = np.bincount(u, weights=(r <= 2).astype(np.float64), minlength=n_users)
    safe_n = np.maximum(n, 1)
    mean_u = np.where(n > 0, s1 / safe_n, gmean)
    std_u = np.sqrt(np.maximum(s2 / safe_n - (s1 / safe_n) ** 2, 0))
    gr_pop = content.meta["log_gr_count"].to_numpy()
    mainstream = np.where(n > 0, np.bincount(u, weights=gr_pop[i], minlength=n_users) / safe_n, gr_pop.mean())

    user_genre = np.asarray(R_like @ content.genre)                                  # суммы по лайкам
    gsum = user_genre.sum(axis=1, keepdims=True)
    user_genre = np.divide(user_genre, gsum, out=np.zeros_like(user_genre), where=gsum > 0)
    ent = -(np.where(user_genre > 0, user_genre * np.log(np.where(user_genre > 0, user_genre, 1)), 0)).sum(axis=1)
    genre_entropy = ent / np.log(content.n_genres)                                   # 0 - один жанр, 1 - всё поровну

    series_share = np.asarray(R_like @ content.has_series.astype(np.float32)).ravel() / np.maximum(n_like, 1)
    a_mat = content.book_entities(content.author_code)
    distinct_authors = np.asarray((R_like @ a_mat > 0).sum(axis=1)).ravel()
    author_repeat = np.where(n_like > 0, 1 - distinct_authors / np.maximum(n_like, 1), 0.0)

    segment = segment_users(n, new_max, active_min)
    user_features = pd.DataFrame({
        "n_ratings": n, "log_n_ratings": np.log1p(n),
        "activity_pct": pd.Series(n).rank(pct=True).to_numpy(),                      # перцентиль активности
        "mean_rating": mean_u, "rating_bias": mean_u - gmean, "rating_std": std_u,
        "like_share": np.where(n > 0, n_like / safe_n, 0.0),
        "dislike_share": np.where(n > 0, n_dis / safe_n, 0.0),
        "mainstream": mainstream,                                                    # любит ли популярные книги
        "genre_entropy": genre_entropy,                                              # широта вкуса
        "series_share": series_share, "author_repeat": author_repeat,
        "segment": segment,
    })

    # ----- признаки книг (по оценкам из train)
    ni = np.bincount(i, minlength=n_items).astype(np.float64)
    si = np.bincount(i, weights=r, minlength=n_items)
    si2 = np.bincount(i, weights=r * r, minlength=n_items)
    li = np.bincount(i, weights=like.astype(np.float64), minlength=n_items)
    safe_ni = np.maximum(ni, 1)
    mean_i = np.where(ni > 0, si / safe_ni, gmean)
    glob_like = like.mean()
    dist = np.stack([np.bincount(i[r == k], minlength=n_items) for k in range(1, 6)], axis=1).astype(np.float64)
    p = dist / np.maximum(dist.sum(axis=1, keepdims=True), 1)
    ent_i = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(axis=1) / np.log(5)
    item_features = pd.DataFrame({
        "n_ratings": ni, "log_n_ratings": np.log1p(ni),
        "popularity_pct": pd.Series(ni).rank(pct=True).to_numpy(),
        "bayes_mean": (si + prior_strength * gmean) / (ni + prior_strength),         # средняя оценка со сжатием
        "rating_std": np.sqrt(np.maximum(si2 / safe_ni - (si / safe_ni) ** 2, 0)),
        "rating_entropy": ent_i,                                                     # разнообразие оценок
        "like_ratio": (li + prior_strength * glob_like) / (ni + prior_strength),
        "main_genre": content.genre.argmax(axis=1),
    })
    item_features = pd.concat([item_features, content.meta], axis=1)
    return FeatureStore(content, user_features, item_features, user_genre.astype(np.float32), R_like, segment)


# --------------------------------------------------------------------------- признаки взаимодействий
def author_series_counts(store: FeatureStore, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой пары (пользователь, книга): сколько понравившихся пользователю книг написал тот же автор /
    входит в ту же серию. Возвращает две плотные матрицы (len(users), n_items)."""
    c = store.content
    likes = store.likes[users]
    out = []
    for codes in (c.author_code, c.series_code):
        ent = c.book_entities(codes)                                 # книги x сущности (one-hot)
        out.append((likes @ ent @ ent.T).toarray().astype(np.float32))
    return out[0], out[1]


def genre_affinity(store: FeatureStore, users: np.ndarray) -> np.ndarray:
    """Косинус между профилем тем пользователя и темами книги."""
    ug = normalize(store.user_genre[users])
    ig = normalize(store.content.genre)
    return (ug @ ig.T).astype(np.float32)


def max_history_similarity(store: FeatureStore, users: np.ndarray, max_hist: int = 40,
                           batch: int = 64) -> np.ndarray:
    """Схожесть книги с самой близкой из понравившихся пользователю (по текстовым эмбеддингам)."""
    e = store.item_emb
    out = np.zeros((len(users), e.shape[0]), dtype=np.float32)
    likes = store.likes
    for s in range(0, len(users), batch):
        for k, u in enumerate(users[s:s + batch]):
            liked = likes.indices[likes.indptr[u]:likes.indptr[u + 1]]
            if len(liked) == 0:
                continue
            liked = liked[:max_hist]
            out[s + k] = (e @ e[liked].T).max(axis=1)
    return out


def interaction_features(store: FeatureStore, users: np.ndarray, items: np.ndarray) -> pd.DataFrame:
    """Таблица признаков взаимодействия для пар (users[k], items[k]) - для анализа и отладки."""
    users, items = np.asarray(users), np.asarray(items)
    c, rows = store.content, []
    ent_a, ent_s = c.book_entities(c.author_code), c.book_entities(c.series_code)
    likes = store.likes[users]
    same_author = np.asarray((likes @ ent_a).multiply(ent_a[items]).sum(axis=1)).ravel()
    same_series = np.asarray((likes @ ent_s).multiply(ent_s[items]).sum(axis=1)).ravel()
    ug, ig = normalize(store.user_genre[users]), normalize(c.genre[items])
    sim = np.zeros(len(users), dtype=np.float32)
    for k, (u, i) in enumerate(zip(users, items)):
        liked = store.likes.indices[store.likes.indptr[u]:store.likes.indptr[u + 1]]
        if len(liked):
            sim[k] = (store.item_emb[liked] @ store.item_emb[i]).max()
    uf = store.user_features["mainstream"].to_numpy()[users]
    return pd.DataFrame({
        "same_author_likes": same_author, "same_series_likes": same_series,
        "genre_affinity": (ug * ig).sum(axis=1), "max_history_sim": sim,
        "pop_gap": np.abs(store.item_features["log_gr_count"].to_numpy()[items] - uf),
        "item_bias_vs_user": store.item_features["bayes_mean"].to_numpy()[items]
        - store.user_features["mean_rating"].to_numpy()[users],
    })
