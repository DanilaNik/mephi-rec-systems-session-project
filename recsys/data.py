"""Загрузка, очистка и разбиение данных Goodbooks-10k.

Код перенесён из домашнего задания №1 (протокол разбиения и нумерация u/i не менялись, поэтому
метрики итогового проекта напрямую сопоставимы с ДЗ).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

LIKE_THRESHOLD = 4  # оценка >= 4 считается «понравилось»

# Служебные теги-«полки» Goodreads: описывают действие читателя, а не содержание книги
SHELF_TAGS = {
    "to-read", "currently-reading", "favorites", "favourites", "favorite", "favourite",
    "all-time-favorites", "books-i-own", "owned", "owned-books", "i-own", "my-books",
    "my-library", "default", "kindle", "library", "to-buy", "ebook", "ebooks", "e-book",
    "e-books", "audiobook", "audiobooks", "audio", "audio-books", "audible", "book-club",
    "read", "finished", "did-not-finish", "dnf", "abandoned", "tbr", "wish-list",
    "wishlist", "borrowed", "own", "have", "owned-but-unread", "unread", "to-read-fiction",
    "my-bookshelf", "books", "bookshelf", "read-more-than-once", "re-read", "reread",
}
SHELF_REGEX = re.compile(r"(read|owned?|shelf).*\d{4}|^\d{4}$|^(read|to-read)-in-")

REQUIRED_COLUMNS = {
    "ratings": {"user_id", "book_id", "rating"},
    "books": {"id", "book_id", "title", "authors", "ratings_count", "average_rating",
              "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"},
    "tags": {"tag_id", "tag_name"},
    "book_tags": {"goodreads_book_id", "tag_id", "count"},
}


def is_shelf_tag(name: str) -> bool:
    return name in SHELF_TAGS or bool(SHELF_REGEX.search(name))


@dataclass
class Dataset:
    ratings: pd.DataFrame      # user_id, book_id, rating, u, i (без дублей)
    books: pd.DataFrame        # индекс = i
    book_tags: pd.DataFrame    # goodreads_book_id, tag_id, count, tag_name
    user_ids: np.ndarray       # u -> user_id
    n_users: int
    n_items: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_users, self.n_items

    def user_index(self, user_id: int) -> int:
        return int(np.searchsorted(self.user_ids, user_id))


def load_raw(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """Читает четыре CSV и проверяет, что в них есть нужные колонки (чтобы пайплайн падал сразу и понятно)."""
    d = Path(data_dir)
    raw = {}
    for name, cols in REQUIRED_COLUMNS.items():
        path = d / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл {path}")
        df = pd.read_csv(path)
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"В {path.name} нет колонок: {sorted(missing)}")
        raw[name] = df
    return raw


def build_dataset(raw: dict[str, pd.DataFrame]) -> Dataset:
    """Очистка данных: дубли оценок, отрицательные count у тегов, переименование ключей books."""
    books = raw["books"].rename(columns={"id": "book_id", "book_id": "goodreads_book_id"})
    books = books.sort_values("book_id").reset_index(drop=True)      # индекс = i

    ratings = raw["ratings"].drop_duplicates(["user_id", "book_id"], keep="last")
    ratings = ratings[ratings["book_id"].isin(books["book_id"])]
    user_ids = np.sort(ratings["user_id"].unique())
    item_of_book = pd.Series(books.index.to_numpy(), index=books["book_id"])
    ratings = ratings.assign(
        u=np.searchsorted(user_ids, ratings["user_id"].to_numpy()),
        i=item_of_book.loc[ratings["book_id"]].to_numpy(),
    ).reset_index(drop=True)

    book_tags = raw["book_tags"]
    book_tags = book_tags[book_tags["count"] > 0].merge(raw["tags"], on="tag_id", how="left")
    return Dataset(ratings, books, book_tags, user_ids, len(user_ids), len(books))


def to_csr(df: pd.DataFrame, shape: tuple[int, int], value: str | None = "rating",
           dtype=np.float32) -> sparse.csr_matrix:
    """Таблица (u, i, rating) -> разреженная матрица пользователь x книга (value=None даёт 0/1)."""
    data = np.ones(len(df), dtype=dtype) if value is None else df[value].to_numpy(dtype)
    return sparse.csr_matrix((data, (df["u"].to_numpy(), df["i"].to_numpy())), shape=shape)


def split_per_user(df: pd.DataFrame, test_frac: float, seed: int = 42):
    """Случайное разбиение внутри каждого пользователя.

    В тест уходит round(test_frac * n) оценок пользователя, минимум одна, но не все.
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    grp = shuffled.groupby("u")["u"]
    n = grp.transform("size").to_numpy()
    pos = grp.cumcount().to_numpy()
    n_test = np.minimum(np.maximum(1, np.round(n * test_frac)), n - 1)
    is_test = pos < n_test
    return shuffled[~is_test].sort_index(), shuffled[is_test].sort_index()


@dataclass
class Splits:
    """train / val / trainval / test: протокол из ДЗ (70% / 10% / 20%, внутри каждого пользователя)."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    trainval_df: pd.DataFrame
    test_df: pd.DataFrame
    R_train: sparse.csr_matrix
    R_trainval: sparse.csr_matrix


def make_splits(ds: Dataset, test_frac: float = 0.20, val_frac_of_trainval: float = 0.125,
                seed: int = 42) -> Splits:
    trainval_df, test_df = split_per_user(ds.ratings, test_frac, seed=seed)
    train_df, val_df = split_per_user(trainval_df, val_frac_of_trainval, seed=seed + 1)
    assert not set(zip(trainval_df.u, trainval_df.i)) & set(zip(test_df.u, test_df.i)), "утечка train -> test!"
    return Splits(train_df, val_df, trainval_df, test_df,
                  to_csr(train_df, ds.shape), to_csr(trainval_df, ds.shape))
