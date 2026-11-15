"""Этап 2 и 3. Гибридная система: взвешенное усреднение моделей, сегменты пользователей, оптимизация весов,
генерация кандидатов и балансировка выдачи (релевантность / разнообразие / популярное).

Схема итоговой системы (класс `HybridSystem`)::

    скоры моделей --z-score--> веса сегмента пользователя --> гибридный скор  (HybridRecommender)
          |                                                          |
          +--- top-N каждой модели ---> единый пул кандидатов -------+--> фильтр прочитанного
                                                                              |
                                              переранжирование MMR (разнообразие) + слоты под популярное
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .features import USER_SEGMENTS
from .metrics import FastEvaluator, Recommender, mask_seen, top_k, zscore


# --------------------------------------------------------------------------- смешивание скоров
def normalize_scores(scores: np.ndarray, clip: float | None = None) -> np.ndarray:
    """z-score по строке (пользователю). Для разреженных скоров (почти все нули) значения у ненулевых
    книг огромны, поэтому по желанию ограничиваем сверху значением clip."""
    z = zscore(np.asarray(scores, dtype=np.float32))
    return np.minimum(z, clip) if clip is not None else z


def blend(z: dict[str, np.ndarray], weights: dict[str, float], conf: np.ndarray,
          cold: dict[str, float] | None = None) -> np.ndarray:
    """score = conf_i * sum_m w_m z_m + (1 - conf_i) * sum_c cold_c z_c.

    z - нормированные скоры моделей; conf_i = n_i / (n_i + c) - доверие к коллаборативным сигналам
    для книги i; cold - веса моделей, которые работают на «холодных» книгах без оценок (контентная).
    """
    cold = cold or {"content": 1.0}
    total = sum(w for w in weights.values() if w > 0)
    mix = sum((w / total) * z[k] for k, w in weights.items() if w > 0)
    cold_total = sum(cold.values())
    fallback = sum((w / cold_total) * z[k] for k, w in cold.items())
    return conf[None, :] * mix + (1 - conf[None, :]) * fallback


def item_confidence(item_counts: np.ndarray, c: float = 3.0) -> np.ndarray:
    return (item_counts / (item_counts + c)).astype(np.float32)


# --------------------------------------------------------------------------- оптимизация весов
@dataclass
class WeightSearch:
    weights: dict[str, float]
    score: float
    history: pd.DataFrame = field(repr=False, default=None)     # все проверенные комбинации


def optimize_weights(parts: list[tuple[dict[str, np.ndarray], FastEvaluator, float]], conf: np.ndarray,
                     names: list[str], cold: dict[str, float] | None = None, metric: str = "nDCG@10",
                     n_random: int = 40, n_sweeps: int = 4, seed: int = 0, min_gain: float = 5e-4,
                     prune: float = 0.03, init: dict[str, float] | None = None) -> WeightSearch:
    """Подбор весов гибрида на val.

    parts - список (z, evaluator, доля): нормированные скоры моделей для выборки пользователей, оценщик для
    этой выборки и её вес в целевой функции. Одна часть - подбор для одного сегмента; несколько частей с долями
    сегментов - подбор общих весов в естественных пропорциях пользователей.

    1) Старт: чистые модели, равные веса, переданный init и n_random случайных точек симплекса (Дирихле).
    2) Покоординатный локальный поиск: по каждому весу пробуем обнулить, уменьшить/увеличить его и
       сдвинуть на шаг; шаг уменьшается после прохода без улучшений.
    3) Защита от подгонки под шум: улучшение принимается, только если оно больше min_gain (порядок
       стандартной ошибки метрики на выборке), а веса меньше prune в конце обнуляются, если это не ухудшает цель.

    Сетка с шагом 0.1 по 6-8 моделям - это 3-20 тысяч комбинаций, поэтому используется поиск.
    Веса нормируются к сумме 1 (на ранжирование масштаб не влияет).
    """
    rng = np.random.default_rng(seed)
    rows, cache = [], {}
    total_share = sum(sh for _, _, sh in parts)

    def f(w: np.ndarray) -> float:
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            return -1.0
        w = w / w.sum()
        key = tuple(np.round(w, 4))
        if key not in cache:
            wd = dict(zip(names, w))
            res = {}
            for z, ev, share in parts:
                for k, v in ev.evaluate(blend(z, wd, conf, cold)).items():
                    res[k] = res.get(k, 0.0) + share / total_share * v
            cache[key] = res[metric]
            rows.append({"eval": len(rows), **wd, **res})
        return cache[key]

    starts = [np.eye(len(names))[k] for k in range(len(names))] + [np.ones(len(names)) / len(names)]
    if init:
        starts.append(np.array([init.get(n, 0.0) for n in names]))
    starts += [rng.dirichlet(np.full(len(names), 0.6)) for _ in range(n_random)]
    best_w = max(starts, key=f)
    best = f(best_w)

    step = 0.2
    for _ in range(n_sweeps):
        improved = False
        for k in rng.permutation(len(names)):
            options = [0.0, best_w[k] * 0.5, best_w[k] * 1.5, best_w[k] + step, max(best_w[k] - step, 0.0)]
            for v in options:
                cand = best_w.copy()
                cand[k] = v
                val = f(cand)
                if val > best + min_gain:
                    best, best_w, improved = val, np.maximum(cand, 0) / np.maximum(cand, 0).sum(), True
        if not improved:
            step /= 2
    pruned = np.where(best_w < prune, 0.0, best_w)
    if pruned.sum() > 0 and f(pruned) >= best - min_gain / 2:
        best_w = pruned / pruned.sum()
        best = f(best_w)
    hist = pd.DataFrame(rows)          # в порядке проверки (колонка eval)
    return WeightSearch({n: float(w) for n, w in zip(names, best_w)}, float(best), hist)


# --------------------------------------------------------------------------- гибридная модель со скорами
class HybridRecommender(Recommender):
    """Взвешенное усреднение нормированных скоров моделей с весами по сегментам пользователей.

    weights - либо один словарь {модель: вес} для всех, либо {сегмент (0/1/2): словарь}.
    segments - сегмент каждого пользователя (по числу оценок в обучающей выборке).
    """
    name = "Hybrid"

    def __init__(self, models: dict[str, Recommender], weights: dict, item_counts: np.ndarray,
                 segments: np.ndarray, cold: dict[str, float] | None = None, c: float = 3.0,
                 clip: dict[str, float] | None = None):
        self.models, self.segments = models, np.asarray(segments)
        self.weights = weights if isinstance(next(iter(weights.values())), dict) else {None: weights}
        self.cold = cold or {"content": 1.0}
        self.conf = item_confidence(item_counts, c)
        self.clip = clip or {}

    @property
    def components(self) -> list[str]:
        used = {k for w in self.weights.values() for k, v in w.items() if v > 0} | set(self.cold)
        return [k for k in self.models if k in used]

    def components_z(self, users: np.ndarray) -> dict[str, np.ndarray]:
        return {k: normalize_scores(self.models[k].score(users), self.clip.get(k)) for k in self.components}

    def blend_z(self, z: dict[str, np.ndarray], users: np.ndarray) -> np.ndarray:
        if None in self.weights:
            return blend(z, self.weights[None], self.conf, self.cold)
        out = np.empty_like(next(iter(z.values())))
        seg = self.segments[users]
        for s, w in self.weights.items():
            rows = np.flatnonzero(seg == s)
            if len(rows):
                out[rows] = blend({k: v[rows] for k, v in z.items()}, w, self.conf, self.cold)
        return out

    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        return self.blend_z(self.components_z(users), users)


# --------------------------------------------------------------------------- пул кандидатов и балансировка
@dataclass
class BalanceConfig:
    """Параметры генерации кандидатов и балансировки выдачи."""
    pool_sources: dict[str, int] = field(default_factory=lambda: {"cf": 50, "content": 30})  # модель -> top-N в пул
    pool_popular: int = 20                       # сколько популярных книг добавить в пул
    mmr_lambda: float = 1.0                      # 1 - только релевантность, меньше - больше разнообразия
    mmr_top: int = 50                            # из скольких лучших по гибридному скору выбираем в MMR
    pop_share: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0})  # доля слотов под популярное


def mmr_select(cand: np.ndarray, rel: np.ndarray, item_emb: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Maximal Marginal Relevance для всех пользователей сразу.

    cand (B, M) - индексы кандидатов, rel (B, M) - их гибридные скоры (-inf, если кандидата нет).
    На каждом шаге выбирается книга с максимумом  lam * rel_norm - (1 - lam) * max_sim_to_selected,
    где rel_norm - скор, приведённый к [0, 1] внутри списка пользователя, а sim - косинус текстовых эмбеддингов.
    """
    b, m = cand.shape
    finite = np.isfinite(rel)
    lo = np.where(finite, rel, np.inf).min(axis=1, keepdims=True)
    hi = np.where(finite, rel, -np.inf).max(axis=1, keepdims=True)
    rel_n = np.where(finite, (rel - lo) / np.maximum(hi - lo, 1e-9), -np.inf)
    v = item_emb[cand]                                              # (B, M, d)
    sims = np.einsum("bmd,bnd->bmn", v, v)
    max_sim = np.zeros((b, m), dtype=np.float32)
    chosen = np.zeros((b, k), dtype=np.int64)
    taken = np.zeros((b, m), dtype=bool)
    rows = np.arange(b)
    for t in range(k):
        obj = lam * rel_n - (1 - lam) * max_sim
        obj[taken] = -np.inf
        j = obj.argmax(axis=1)
        chosen[:, t] = j
        taken[rows, j] = True
        max_sim = np.maximum(max_sim, sims[rows, j])
    return np.take_along_axis(cand, chosen, axis=1)


class HybridSystem(Recommender):
    """Итоговая система рекомендаций.

    1. Гибридный скор (веса зависят от сегмента пользователя) для всех книг.
    2. Единый пул кандидатов: top-N каждой выбранной модели плюс популярные книги; прочитанное исключено.
    3. Переранжирование пула: MMR (баланс релевантности и разнообразия) и слоты под популярное.
    4. Пользователь без истории получает неперсональную выдачу (популярное).
    """
    name = "Hybrid system"

    def __init__(self, hybrid: HybridRecommender, item_emb: np.ndarray, pop_scores: np.ndarray,
                 balance: BalanceConfig | None = None):
        self.hybrid, self.item_emb, self.pop_scores = hybrid, item_emb, np.asarray(pop_scores, dtype=np.float32)
        self.balance = balance or BalanceConfig()
        self.n_history = None            # заполняется через set_history (число оценок пользователей)

    def set_history(self, train: sparse.csr_matrix) -> "HybridSystem":
        self.n_history = np.asarray(train.getnnz(axis=1)).ravel()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        """Гибридные скоры, ограниченные пулом кандидатов (вне пула -inf); прочитанное не маскируется."""
        users = np.asarray(users)
        z = self.hybrid.components_z(users)
        return np.where(self._pool(z, None), self.hybrid.blend_z(z, users), -np.inf)

    def pool_mask(self, users: np.ndarray, seen: sparse.csr_matrix | None = None) -> np.ndarray:
        """Булева матрица (пользователи x книги): входит ли книга в единый пул кандидатов."""
        return self._pool(self.hybrid.components_z(np.asarray(users)), seen)

    def _pool(self, z: dict[str, np.ndarray], seen: sparse.csr_matrix | None) -> np.ndarray:
        b, n_items = next(iter(z.values())).shape
        pool = np.zeros((b, n_items), dtype=bool)
        rows = np.arange(b)[:, None]
        for src, n in self.balance.pool_sources.items():
            if src in z:
                zz = np.array(z[src], dtype=np.float32)
                mask_seen(zz, seen)
                pool[rows, top_k(zz, n)] = True
        if self.balance.pool_popular:
            pop = np.broadcast_to(self.pop_scores, (b, n_items)).copy()
            mask_seen(pop, seen)
            pool[rows, top_k(pop, self.balance.pool_popular)] = True
        return pool

    def rank(self, users: np.ndarray, k: int, seen: sparse.csr_matrix | None = None) -> np.ndarray:
        users = np.asarray(users)
        bal = self.balance
        z = self.hybrid.components_z(users)
        s = np.array(self.hybrid.blend_z(z, users), dtype=np.float32)
        mask_seen(s, seen)
        s = np.where(self._pool(z, seen), s, -np.inf)

        # пользователи без истории: неперсональная выдача
        cold = np.zeros(len(users), dtype=bool) if self.n_history is None else self.n_history[users] == 0
        if cold.any():
            pop = np.broadcast_to(self.pop_scores, (cold.sum(), s.shape[1])).copy()
            mask_seen(pop, None if seen is None else seen[np.flatnonzero(cold)])
            s[cold] = pop

        shares = np.array([bal.pop_share.get(int(g), 0.0) for g in self.hybrid.segments[users]])
        n_pop = np.where(cold, 0, np.round(shares * k).astype(int))
        k_pers = k - n_pop
        m = max(bal.mmr_top, k)
        cand = top_k(s, m)
        rel = np.take_along_axis(s, cand, axis=1)
        if bal.mmr_lambda < 1.0:
            recs = mmr_select(cand, rel, self.item_emb, k, bal.mmr_lambda)
        else:
            recs = cand[:, :k]
        if n_pop.any():
            recs = self._insert_popular(recs, n_pop, seen)
        return recs

    def _insert_popular(self, recs: np.ndarray, n_pop: np.ndarray, seen: sparse.csr_matrix | None) -> np.ndarray:
        """Последние n_pop позиций выдачи заменяются самыми популярными непрочитанными книгами."""
        order = np.argsort(-self.pop_scores, kind="stable")[:max(200, 4 * recs.shape[1])]
        out = recs.copy()
        k = recs.shape[1]
        for r in np.flatnonzero(n_pop):
            keep = list(recs[r, :k - n_pop[r]])
            seen_r = set() if seen is None else set(seen.indices[seen.indptr[r]:seen.indptr[r + 1]])
            extra = [i for i in order if i not in seen_r and i not in keep][:n_pop[r]]
            out[r] = keep + extra
        return out
