"""Метрики качества и общий интерфейс рекомендателей.

Precision@K, Recall@K, nDCG@K и вспомогательные функции перенесены из ДЗ. Добавлены метрики
разнообразия (ILD, novelty, coverage) и быстрый оценщик `FastEvaluator` для подбора весов гибрида.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .data import LIKE_THRESHOLD, to_csr


# --------------------------------------------------------------------------- базовые операции со скорами
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k лучших элементов в каждой строке (по убыванию скора)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]          # O(n), без полной сортировки
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def mask_seen(scores: np.ndarray, seen: sparse.spmatrix | None) -> np.ndarray:
    """Ставит -inf для книг, которые пользователь уже оценил (меняет массив на месте)."""
    if seen is not None:
        coo = seen.tocoo()
        scores[coo.row, coo.col] = -np.inf
    return scores


def zscore(scores: np.ndarray) -> np.ndarray:
    """Стандартизация скоров каждого пользователя (по строкам)."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.where(std > 0, std, 1.0)


# --------------------------------------------------------------------------- метрики ранжирования
def ranking_metrics(hits: np.ndarray, n_relevant: np.ndarray, ks: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Precision@K, Recall@K и nDCG@K для каждого пользователя.

    hits[u, r] - релевантна ли книга на позиции r (с нуля) в выдаче пользователя u,
    n_relevant[u] - число релевантных книг пользователя в тесте (больше нуля).
    """
    hits = hits.astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2))      # 1 / log2(позиция + 1)
    ideal_cum = np.cumsum(discounts)
    out = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(axis=1)
        dcg = (h * discounts[:k]).sum(axis=1)
        idcg = ideal_cum[np.minimum(k, n_relevant) - 1]             # идеальная выдача
        out[f"P@{k}"] = n_hit / k
        out[f"R@{k}"] = n_hit / n_relevant
        out[f"nDCG@{k}"] = dcg / idcg
    return out


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)) ** 2)))


# --------------------------------------------------------------------------- метрики разнообразия
def list_diversity(recs: np.ndarray, ks: tuple[int, ...], item_pop: np.ndarray, n_users_train: int,
                   n_cand: int, item_emb: np.ndarray | None = None) -> dict[str, float]:
    """Метрики выдач целиком (не зависят от релевантности).

    coverage@K - доля каталога, попавшая хоть в одну выдачу;
    avg_pop@K - среднее число train-оценок у рекомендованных книг;
    novelty@K - средняя «информативность» книги -log2(p_i), p_i = (n_i + 1) / (n_users + 1): чем больше, тем реже книги;
    ild@K - Intra-List Diversity: средняя попарная (1 - косинус) эмбеддингов книг внутри выдачи.
    """
    out = {}
    self_info = -np.log2((item_pop + 1.0) / (n_users_train + 1.0))
    for k in ks:
        r = recs[:, :k]
        out[f"coverage@{k}"] = len(np.unique(r)) / n_cand
        out[f"avg_pop@{k}"] = float(item_pop[r].mean())
        out[f"novelty@{k}"] = float(self_info[r].mean())
        if item_emb is not None and k > 1:
            v = item_emb[r]                                           # (users, k, d), строки нормированы
            gram = np.einsum("ukd,ujd->ukj", v, v)
            out[f"ild@{k}"] = float(((k * k - gram.sum(axis=(1, 2))) / (k * (k - 1))).mean())
    return out


# --------------------------------------------------------------------------- интерфейс моделей
class Recommender(ABC):
    name: str = "model"

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Матрица скоров (len(users), n_items): чем больше, тем выше книга в выдаче."""

    def recommend(self, users: np.ndarray, n: int = 10, seen: sparse.csr_matrix | None = None) -> np.ndarray:
        """Индексы top-n книг, которых пользователь ещё не оценил."""
        users = np.asarray(users)
        scores = np.array(self.score(users), dtype=np.float32)
        mask_seen(scores, None if seen is None else seen[users])
        return top_k(scores, n)


class PrecomputedScores(Recommender):
    """Отдаёт заранее посчитанную матрицу скоров (для быстрого перебора весов)."""
    name = "precomputed"

    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def score(self, users: np.ndarray) -> np.ndarray:
        assert len(users) == len(self.scores)
        return self.scores


@dataclass
class EvalResult:
    summary: pd.Series      # метрики, усреднённые по пользователям
    per_user: pd.DataFrame  # метрики каждого пользователя
    recs: np.ndarray | None = None   # выдачи (строки соответствуют per_user.index)


def evaluate_ranking(model, train: sparse.csr_matrix, test_df: pd.DataFrame,
                     ks: tuple[int, ...] = (5, 10, 20), threshold: float = LIKE_THRESHOLD,
                     users: np.ndarray | None = None, candidates: np.ndarray | None = None,
                     batch_size: int = 2000, item_emb: np.ndarray | None = None) -> EvalResult:
    """Считает Precision/Recall/nDCG@K (+ coverage, avg_pop, novelty и ild, если передан item_emb).

    Релевантные книги пользователя - его оценки в test_df не ниже threshold. Кандидаты -
    все книги, которых нет в train (или только candidates). Усредняем по пользователям,
    у которых есть хотя бы одна релевантная книга.

    Если у модели есть метод `rank(users, k, seen)`, возвращающий готовые выдачи (так устроена
    итоговая гибридная система с пулом кандидатов и переранжированием), используется он,
    иначе выдача строится по `score`.
    """
    n_users, n_items = train.shape
    rel = to_csr(test_df[test_df["rating"] >= threshold], train.shape, value=None, dtype=np.int8)
    rel_counts = np.asarray(rel.sum(axis=1)).ravel()
    eval_users = np.flatnonzero(rel_counts > 0)
    if users is not None:
        eval_users = np.intersect1d(eval_users, users)

    allowed = None
    if candidates is not None:
        allowed = np.zeros(n_items, dtype=bool)
        allowed[candidates] = True

    k_max = max(ks)
    recs, metrics = [], {}
    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start:start + batch_size]
        if hasattr(model, "rank") and allowed is None:
            rec = model.rank(batch, k_max, train[batch])
        else:
            scores = np.array(model.score(batch), dtype=np.float32)    # копия, дальше пишем в неё
            mask_seen(scores, train[batch])
            if allowed is not None:
                scores[:, ~allowed] = -np.inf
            rec = top_k(scores, k_max)
        hits = np.take_along_axis(rel[batch].toarray(), rec, axis=1).astype(bool)
        for name, values in ranking_metrics(hits, rel_counts[batch], ks).items():
            metrics.setdefault(name, []).append(values)
        recs.append(rec)

    recs = np.vstack(recs)
    per_user = pd.DataFrame({k: np.concatenate(v) for k, v in metrics.items()}, index=eval_users)
    per_user.index.name = "u"
    per_user["n_train"] = np.asarray(train[eval_users].getnnz(axis=1)).ravel()

    summary = per_user.drop(columns="n_train").mean()
    item_pop = np.asarray(train.getnnz(axis=0)).ravel()
    n_cand = n_items if candidates is None else len(candidates)
    for name, value in list_diversity(recs, ks, item_pop, n_users, n_cand, item_emb).items():
        summary[name] = value
    summary["n_users"] = len(eval_users)
    return EvalResult(summary, per_user, recs)


def results_table(results: dict[str, EvalResult], columns: list[str]) -> pd.DataFrame:
    """Сводная таблица: строки - модели, колонки - выбранные метрики."""
    return pd.DataFrame({name: r.summary for name, r in results.items()}).T[columns]


def paired_diff(a: EvalResult, b: EvalResult, metric: str) -> pd.Series:
    """Разность метрики по пользователям (парное сравнение) и 95% доверительный интервал."""
    idx = a.per_user.index.intersection(b.per_user.index)
    d = a.per_user.loc[idx, metric] - b.per_user.loc[idx, metric]
    m, se = d.mean(), d.std() / np.sqrt(len(d))
    return pd.Series({"разность": m, "95% ДИ от": m - 1.96 * se, "95% ДИ до": m + 1.96 * se,
                      "значимо": bool(m - 1.96 * se > 0 or m + 1.96 * se < 0)})


# --------------------------------------------------------------------------- быстрый оценщик для подбора весов
class FastEvaluator:
    """Считает nDCG/Recall/Precision@K для заранее посчитанных матриц скоров фиксированной выборки пользователей.

    Маски «уже оценено» и релевантные книги хранятся плотными булевыми матрицами, поэтому одна оценка -
    это np.where + argpartition без обращений к разреженным матрицам. Это нужно, потому что при подборе
    весов гибрида оценка вызывается сотни раз.
    """

    def __init__(self, train: sparse.csr_matrix, eval_df: pd.DataFrame, users: np.ndarray, k: int = 10,
                 threshold: float = LIKE_THRESHOLD):
        rel = to_csr(eval_df[eval_df["rating"] >= threshold], train.shape, value=None, dtype=np.int8)
        users = np.asarray(users)
        users = users[np.asarray(rel[users].sum(axis=1)).ravel() > 0]
        self.users, self.k = users, k
        self.seen = train[users].toarray() > 0
        self.rel = rel[users].toarray() > 0
        self.n_rel = self.rel.sum(axis=1)
        self.discounts = 1.0 / np.log2(np.arange(2, k + 2))
        self.idcg = np.cumsum(self.discounts)[np.minimum(k, self.n_rel) - 1]

    def evaluate(self, scores: np.ndarray) -> dict[str, float]:
        assert scores.shape == self.seen.shape
        masked = np.where(self.seen, -np.inf, scores)
        rec = top_k(masked, self.k)
        hits = np.take_along_axis(self.rel, rec, axis=1)
        n_hit = hits.sum(axis=1)
        dcg = (hits * self.discounts).sum(axis=1)
        return {f"nDCG@{self.k}": float((dcg / self.idcg).mean()),
                f"R@{self.k}": float((n_hit / self.n_rel).mean()),
                f"P@{self.k}": float((n_hit / self.k).mean())}
