"""Модели из домашнего задания №1 (Popularity, Content-based, Item-based CF, SVD, PureSVD)
и две лёгкие модели на расширенных признаках (AuthorSeriesAffinity, ItemQualityPrior).

Все модели наследуются от `Recommender` и умеют одно: `score(users)` -> матрица скоров (пользователи x книги).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from .data import LIKE_THRESHOLD
from .features import FeatureStore, ItemContent, author_series_counts
from .metrics import Recommender


# --------------------------------------------------------------------------- неперсонализированные
class PopularityRecommender(Recommender):
    """Одинаковый топ для всех.

    mean_rating - средняя оценка среди книг, у которых не меньше min_ratings оценок;
    bayesian - взвешенный рейтинг как в IMDb: (v*R + m*C) / (v + m);
    count - число оценок.
    """

    def __init__(self, strategy: str = "mean_rating", min_ratings: int = 50):
        assert strategy in {"mean_rating", "bayesian", "count"}
        self.strategy, self.min_ratings = strategy, min_ratings
        self.name = f"Popularity ({strategy})"

    def fit(self, train: sparse.csr_matrix) -> "PopularityRecommender":
        counts = np.asarray(train.getnnz(axis=0)).ravel().astype(np.float64)
        sums = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
        self.counts_ = counts
        self.means_ = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        if self.strategy == "mean_rating":
            self.scores_ = np.where(counts >= self.min_ratings, self.means_, 0.0)   # 0 меньше любой оценки
        elif self.strategy == "bayesian":
            c_global = sums.sum() / counts.sum()
            self.scores_ = (sums + self.min_ratings * c_global) / (counts + self.min_ratings)
        else:
            self.scores_ = counts
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.scores_.astype(np.float32), (len(users), len(self.scores_)))

    def top_n(self, n: int, books: pd.DataFrame) -> pd.DataFrame:
        idx = np.argsort(-self.scores_, kind="stable")[:n]
        out = books.loc[idx, ["book_id", "title", "authors"]].copy()
        out["mean_rating"] = self.means_[idx]
        out["n_ratings"] = self.counts_[idx].astype(int)
        return out.reset_index(drop=True)


class RandomRecommender(Recommender):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: sparse.csr_matrix) -> "RandomRecommender":
        self.n_items = train.shape[1]
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.random.default_rng(self.seed).random((len(users), self.n_items), dtype=np.float32)


class ItemQualityPrior(Recommender):
    """Неперсональный «приор качества» книги на признаках: смесь стандартизованных
    Байесовской средней оценки в train и глобальной средней оценки Goodreads.

    В ДЗ роль приора неявно играла компонента SVD (смещение книги b_i). Здесь он задан явно.
    """
    name = "Item quality prior"

    def __init__(self, w_train: float = 0.5):
        self.w_train = w_train

    def fit(self, store: FeatureStore) -> "ItemQualityPrior":
        f = store.item_features
        std = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        self.scores_ = (self.w_train * std(f["bayes_mean"]) + (1 - self.w_train) * std(f["gr_avg_rating"])
                        ).to_numpy(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.scores_, (len(users), len(self.scores_)))


# --------------------------------------------------------------------------- контентная модель
class ContentBasedRecommender(Recommender):
    """TF-IDF по профилям книг. Профиль пользователя - сумма векторов понравившихся книг."""

    name = "Content-based (TF-IDF)"

    def __init__(self, like_threshold: float = LIKE_THRESHOLD):
        self.like_threshold = like_threshold

    def fit_items(self, content: ItemContent, books: pd.DataFrame) -> "ContentBasedRecommender":
        self.books, self.tfidf = books, content.tfidf
        return self

    def fit(self, train: sparse.csr_matrix) -> "ContentBasedRecommender":
        self.likes_ = (train >= self.like_threshold).astype(np.float32).tocsr()
        return self

    def get_similar_books(self, book_id: int, N: int = 5) -> pd.DataFrame:
        """N книг, ближайших к book_id по косинусу TF-IDF-векторов."""
        item = int(self.books.index[self.books["book_id"] == book_id][0])
        sims = (self.tfidf @ self.tfidf[item].T).toarray().ravel()
        sims[item] = -np.inf
        idx = np.argsort(-sims, kind="stable")[:N]
        out = self.books.loc[idx, ["book_id", "title", "authors"]].copy()
        out["similarity"] = sims[idx]
        return out.reset_index(drop=True)

    def score(self, users: np.ndarray) -> np.ndarray:
        profile = (self.likes_[users] @ self.tfidf).toarray()
        norm = np.linalg.norm(profile, axis=1, keepdims=True)
        profile /= np.where(norm > 0, norm, 1.0)
        return np.asarray(self.tfidf @ profile.T).T


# --------------------------------------------------------------------------- item-based CF
class ItemBasedCF(Recommender):
    """Item-based CF: top-K соседей для каждой книги, shrinkage, предсказание оценок с демпфированием.

    feedback="explicit": вектор книги - оценки, центрированные по среднему пользователя
    (косинус таких векторов - adjusted cosine); используется для предсказания оценок.
    feedback="implicit": вектор книги - 1, если оценка >= порога, иначе 0; используется для top-N.
    """

    def __init__(self, k: int = 50, feedback: str = "implicit", shrinkage: float = 10.0,
                 like_threshold: float = LIKE_THRESHOLD, damping: float = 0.0, chunk: int = 500):
        assert feedback in {"explicit", "implicit"}
        self.k, self.feedback, self.shrinkage = k, feedback, shrinkage
        self.like_threshold, self.damping, self.chunk = like_threshold, damping, chunk
        self.name = f"Item-based CF ({feedback}, K={k})"

    def fit(self, train: sparse.csr_matrix) -> "ItemBasedCF":
        train = train.tocsr().astype(np.float32)
        self.n_items = train.shape[1]
        rated = (train != 0).astype(np.float32).tocsr()

        counts = np.asarray(rated.sum(axis=1)).ravel()
        sums = np.asarray(train.sum(axis=1)).ravel()
        self.user_mean_ = np.divide(sums, counts, out=np.full(len(sums), train.data.mean()),
                                    where=counts > 0).astype(np.float32)
        centred = train.copy()
        centred.data = centred.data - np.repeat(self.user_mean_, np.diff(train.indptr))

        n_i = np.asarray(rated.sum(axis=0)).ravel()
        self.item_dev_ = np.divide(np.asarray(centred.sum(axis=0)).ravel(), n_i,
                                   out=np.zeros(self.n_items), where=n_i > 0).astype(np.float32)

        resid = train.copy()
        row = np.repeat(np.arange(train.shape[0]), np.diff(train.indptr))
        resid.data = train.data - self.user_mean_[row] - self.item_dev_[train.indices]
        self.resid_ = resid

        if self.feedback == "explicit":
            x, support = centred, rated
        else:
            x = (train >= self.like_threshold).astype(np.float32).tocsr()
            support = x
        self.x_, self.rated_ = x.tocsr(), rated
        self.S_ = self._top_k_similarities(self.x_, support.tocsr())
        return self

    def _top_k_similarities(self, x: sparse.csr_matrix, support: sparse.csr_matrix) -> sparse.csr_matrix:
        """Схожести книг считаем блоками по chunk книг и оставляем только K лучших соседей."""
        n_items, k = self.n_items, min(self.k, self.n_items - 1)
        xt, st = x.T.tocsr(), support.T.tocsr()
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel()) + 1e-9

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.chunk):
            sl = slice(start, min(start + self.chunk, n_items))
            dots = (xt[sl] @ x).toarray()
            common = (st[sl] @ support).toarray()
            sim = dots / (norms[sl, None] * norms[None, :])
            sim *= common / (common + self.shrinkage + 1e-9)
            sim[np.arange(sl.stop - sl.start), np.arange(sl.start, sl.stop)] = 0
            sim[sim < 0] = 0
            nb = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            nb_val = np.take_along_axis(sim, nb, axis=1)
            keep = nb_val > 0
            rows.append(np.repeat(np.arange(sl.start, sl.stop), k).reshape(-1, k)[keep])
            cols.append(nb[keep])
            vals.append(nb_val[keep])
        return sparse.csr_matrix((np.concatenate(vals).astype(np.float32),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n_items, n_items))

    def score(self, users: np.ndarray) -> np.ndarray:
        """Скор книги i для пользователя u: сумма схожестей i с понравившимися книгами среди её K соседей."""
        return (self.x_[users] @ self.S_.T).toarray()

    def predict_ratings(self, users: np.ndarray, items: np.ndarray, damping: float | None = None,
                        batch: int = 50_000) -> np.ndarray:
        """r_hat(u, i) = b_ui + sum_j s_ij * (r_uj - b_uj) / (sum_j s_ij + damping)."""
        damping = self.damping if damping is None else damping
        users, items = np.asarray(users), np.asarray(items)
        preds = np.empty(len(users), dtype=np.float64)
        n_with_neighbours = 0
        for s in range(0, len(users), batch):
            u, i = users[s:s + batch], items[s:s + batch]
            neigh = self.S_[i]
            num = np.asarray(neigh.multiply(self.resid_[u]).sum(axis=1)).ravel()
            den = np.asarray(neigh.multiply(self.rated_[u]).sum(axis=1)).ravel()
            has = den > 0
            base = self.user_mean_[u] + self.item_dev_[i]
            preds[s:s + batch] = base + np.where(has, num / (den + damping + 1e-9), 0.0)
            n_with_neighbours += has.sum()
        self.coverage_ = n_with_neighbours / max(len(users), 1)
        return np.clip(preds, 1, 5)


# --------------------------------------------------------------------------- SVD (surprise) и PureSVD
class SVDRecommender(Recommender):
    """Обёртка над surprise.SVD: r_hat(u, i) = mu + b_u + b_i + p_u * q_i."""

    name = "SVD (surprise)"

    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005,
                 reg_all: float = 0.02, random_state: int = 42):
        self.params = dict(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
                           reg_all=reg_all, random_state=random_state)

    def fit(self, train_df: pd.DataFrame, shape: tuple[int, int]) -> "SVDRecommender":
        from surprise import SVD, Dataset as SurpriseDataset, Reader   # импорт здесь: surprise нужен только этой модели
        reader = Reader(rating_scale=(1, 5))
        trainset = SurpriseDataset.load_from_df(train_df[["u", "i", "rating"]], reader).build_full_trainset()
        algo = SVD(**self.params)
        algo.fit(trainset)
        self.algo, self.trainset = algo, trainset

        n_users, n_items = shape
        f = self.params["n_factors"]
        self.mu = trainset.global_mean
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.P, self.Q = np.zeros((n_users, f), dtype=np.float32), np.zeros((n_items, f), dtype=np.float32)
        raw_u = np.array([trainset.to_raw_uid(k) for k in range(trainset.n_users)])
        raw_i = np.array([trainset.to_raw_iid(k) for k in range(trainset.n_items)])
        self.bu[raw_u], self.P[raw_u] = algo.bu, algo.pu
        self.bi[raw_i], self.Q[raw_i] = algo.bi, algo.qi
        return self

    def predict_ratings(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        raw = self.mu + self.bu[users] + self.bi[items] + np.einsum("nf,nf->n", self.P[users], self.Q[items])
        return np.clip(raw, 1, 5)

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.mu + self.bu[users, None] + self.bi[None, :] + self.P[users] @ self.Q.T).astype(np.float32)


class PureSVDRecommender(Recommender):
    """Усечённое SVD бинарной матрицы лайков X ~ U S V^T. Скор пользователя - проекция x_u V V^T."""

    def __init__(self, n_factors: int = 50, feedback: str = "like",
                 like_threshold: float = LIKE_THRESHOLD, random_state: int = 42):
        assert feedback in {"like", "rated"}
        self.n_factors, self.feedback = n_factors, feedback
        self.like_threshold, self.random_state = like_threshold, random_state
        self.name = f"PureSVD ({feedback}, f={n_factors})"

    def fit(self, train: sparse.csr_matrix) -> "PureSVDRecommender":
        thr = self.like_threshold if self.feedback == "like" else 1
        self.x_ = (train >= thr).astype(np.float32).tocsr()
        k = int(min(self.n_factors, min(self.x_.shape) - 1))
        rng = np.random.default_rng(self.random_state)
        _, s, vt = svds(self.x_, k=k, v0=rng.random(min(self.x_.shape)))
        order = np.argsort(-s)
        self.V_ = np.ascontiguousarray(vt[order].T, dtype=np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.x_[users] @ self.V_) @ self.V_.T


# --------------------------------------------------------------------------- модель на признаках взаимодействий
class AuthorSeriesAffinity(Recommender):
    """Признаки взаимодействия «тот же автор / та же серия, что у понравившихся книг пользователя».

    В EDA ДЗ у 39% понравившихся в test книг тот же автор, что у понравившейся в train (у случайной 1.5%),
    у 23% - та же серия. Скор = w_s * log(1 + число лайков той же серии) + w_a * log(1 + число лайков того же автора).
    """
    name = "Author/series affinity"

    def __init__(self, w_series: float = 1.0, w_author: float = 1.0):
        self.w_series, self.w_author = w_series, w_author

    def fit(self, store: FeatureStore) -> "AuthorSeriesAffinity":
        self.store = store
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        by_author, by_series = author_series_counts(self.store, users)
        return self.w_series * np.log1p(by_series) + self.w_author * np.log1p(by_author)
