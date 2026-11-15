"""Сквозной пайплайн: загрузка данных -> предобработка и признаки -> обучение моделей -> подбор гибрида ->
генерация рекомендаций -> оценка качества -> экспорт результатов.

Каждый этап - отдельный метод `Pipeline`, поэтому их можно запускать по одному (как в ноутбуке) или целиком
(`Pipeline(cfg).run()`, `python run_pipeline.py --data-dir ...`). Весь процесс параметризуется `PipelineConfig`
и полностью воспроизводим (фиксированный seed), так что его можно перезапустить на новых данных: достаточно
указать другую папку с CSV в том же формате.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LIKE_THRESHOLD, Dataset, Splits, build_dataset, load_raw, make_splits, to_csr
from .features import USER_SEGMENTS, FeatureStore, ItemContent, build_features, build_item_content
from .hybrid import (BalanceConfig, HybridRecommender, HybridSystem, item_confidence, normalize_scores,
                     optimize_weights)
from .metrics import (FastEvaluator, PrecomputedScores, evaluate_ranking, paired_diff, results_table, rmse)
from .models import (AuthorSeriesAffinity, ContentBasedRecommender, ItemBasedCF, ItemQualityPrior,
                     PopularityRecommender, PureSVDRecommender, RandomRecommender, SVDRecommender)
from .neural import TwoTowerConfig, TwoTowerRecommender

# гиперпараметры, подобранные на val в домашнем задании №1
DEFAULT_PARAMS = {
    "popularity": {"strategy": "mean_rating", "min_ratings": 40},
    "item_cf": {"k": 10, "shrinkage": 0.0},
    "svd": {"n_factors": 20, "n_epochs": 30, "reg_all": 0.1, "random_state": 42},
    "pure_svd": {"n_factors": 800},
}

MODEL_TITLES = {
    "random": "Random (reference)", "pop": "Popularity (mean rating)", "content": "Content-based (TF-IDF)",
    "cf": "Item-based CF (implicit)", "svd": "SVD (surprise)", "pure": "PureSVD (implicit)",
    "meta": "Author/series affinity", "prior": "Item quality prior", "neural": "Two-Tower (neural)",
}
SEGMENT_TITLES = {0: "новые (<5 оценок)", 1: "обычные (5-29)", 2: "активные (30+)"}


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    out_dir: str = "artifacts/run"
    seed: int = 42
    test_frac: float = 0.20
    val_frac: float = 0.125                  # доля val внутри trainval (0.8 * 0.125 = 10% от всех оценок)
    ks: tuple = (5, 10, 20)
    tune_models: bool = False                # True: подбор гиперпараметров моделей по сетке на val (долго)
    params: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_PARAMS))
    use_neural: bool = True
    neural: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    n_topics: int = 20                       # тематических категорий (NMF)
    emb_dim: int = 64                        # размерность текстовых эмбеддингов книг
    conf_c: float = 3.0                      # параметр доверия к коллаборативному сигналу книги: n / (n + c)
    tune_users_per_segment: int = 3000       # пользователей val для подбора весов в каждом сегменте
    search_random: int = 30                  # случайных стартовых точек при подборе весов
    search_sweeps: int = 4
    pool_sources: dict = field(default_factory=lambda: {"cf": 50, "pure": 30, "content": 30, "meta": 20, "neural": 30})
    pool_popular: int = 20
    mmr_grid: tuple = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    mmr_tolerance: float = 0.02              # допустимая потеря nDCG@10 (в долях) ради разнообразия
    pop_share_grid: tuple = (0.0, 0.1, 0.2)
    n_export_users: int = 2000
    n_recs: int = 10

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))


def synthetic_dataset(data_dir: str | Path, n_users: int = 300, n_books: int = 150, seed: int = 0) -> None:
    """Небольшой синтетический набор в формате Goodbooks-10k (для тестов и демонстрации перезапуска пайплайна)."""
    rng = np.random.default_rng(seed)
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    genres = ["fantasy", "romance", "mystery", "science-fiction", "history", "horror"]
    book_genre = rng.integers(0, len(genres), n_books)
    books = pd.DataFrame({
        "id": np.arange(1, n_books + 1), "book_id": 1000 + np.arange(n_books),
        "title": [f"Book {i} (Series {i // 5}, #{i % 5 + 1})" if i % 3 == 0 else f"Book {i}" for i in range(n_books)],
        "original_title": [f"Book {i}" if i % 7 else None for i in range(n_books)],
        "authors": [f"Author {i // 4}" for i in range(n_books)],
        "original_publication_year": rng.integers(1950, 2017, n_books).astype(float),
        "ratings_count": rng.integers(2000, 200000, n_books), "average_rating": rng.uniform(3.2, 4.6, n_books),
    })
    for r in range(1, 6):
        books[f"ratings_{r}"] = rng.integers(10, 1000, n_books)
    tags = pd.DataFrame({"tag_id": np.arange(len(genres) + 3),
                         "tag_name": genres + ["to-read", "favorites", "fiction"]})
    bt = [(1000 + i, book_genre[i], int(rng.integers(50, 500))) for i in range(n_books)]
    bt += [(1000 + i, len(genres) + int(rng.integers(0, 3)), int(rng.integers(100, 900))) for i in range(n_books)]
    book_tags = pd.DataFrame(bt, columns=["goodreads_book_id", "tag_id", "count"])
    taste = rng.dirichlet(np.ones(len(genres)) * 0.5, n_users)
    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(2.2, 0.8), 3, 80))
        p = taste[u][book_genre] + 0.02
        items = rng.choice(n_books, min(n, n_books), replace=False, p=p / p.sum())
        for i in items:
            rows.append((u + 1, i + 1, int(np.clip(round(3.2 + 3 * taste[u][book_genre[i]] + rng.normal(0, 0.8)), 1, 5))))
    pd.DataFrame(rows, columns=["user_id", "book_id", "rating"]).to_csv(d / "ratings.csv", index=False)
    books.to_csv(d / "books.csv", index=False)
    tags.to_csv(d / "tags.csv", index=False)
    book_tags.to_csv(d / "book_tags.csv", index=False)


# --------------------------------------------------------------------------- построение компонент
def build_components(cfg: PipelineConfig, ds: Dataset, train_df: pd.DataFrame, R, store: FeatureStore,
                     neural_cfg: TwoTowerConfig | None = None, evaluator=None, verbose: bool = False) -> dict:
    """Обучает все модели на train_df (колонки u, i, rating); R и store построены по тем же данным."""
    p = cfg.params
    content = store.content
    models = {
        "random": RandomRecommender().fit(R),
        "pop": PopularityRecommender(**p["popularity"]).fit(R),
        "content": ContentBasedRecommender().fit_items(content, ds.books).fit(R),
        "cf": ItemBasedCF(feedback="implicit", **p["item_cf"]).fit(R),
        "pure": PureSVDRecommender(**p["pure_svd"]).fit(R),
        "meta": AuthorSeriesAffinity().fit(store),
        "prior": ItemQualityPrior().fit(store),
    }
    try:
        models["svd"] = SVDRecommender(**p["svd"]).fit(train_df, ds.shape)
    except ImportError:
        warnings.warn("scikit-surprise не установлен: компонента SVD пропущена")
    if cfg.use_neural:
        models["neural"] = TwoTowerRecommender(neural_cfg or cfg.neural).fit(R, store, evaluator=evaluator, verbose=verbose)
    return models


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None, verbose: bool = True):
        self.cfg = cfg or PipelineConfig()
        self.verbose = verbose
        self.timings: dict[str, float] = {}
        self.rng = np.random.default_rng(self.cfg.seed)

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        if self.verbose:
            print(f"[pipeline] {name} ...", flush=True)
        yield
        self.timings[name] = time.time() - t0
        if self.verbose:
            print(f"[pipeline] {name}: {self.timings[name]:.1f} c", flush=True)

    # ------------------------------------------------------------------ 1-2. данные и признаки
    def load(self) -> "Pipeline":
        with self.stage("1. загрузка и очистка данных"):
            self.raw = load_raw(self.cfg.data_dir)
            self.ds = build_dataset(self.raw)
        return self

    def prepare(self) -> "Pipeline":
        cfg = self.cfg
        with self.stage("2. разбиение и расширенные признаки"):
            self.splits: Splits = make_splits(self.ds, cfg.test_frac, cfg.val_frac, cfg.seed)
            self.content: ItemContent = build_item_content(self.ds, cfg.n_topics, cfg.emb_dim, cfg.seed)
            self.store_train = build_features(self.ds, self.splits.train_df, self.content)         # для подбора на val
            self.store_tv = build_features(self.ds, self.splits.trainval_df, self.content)         # для финальной модели
        return self

    # ------------------------------------------------------------------ 3. обучение
    def tune_hyperparams(self) -> "Pipeline":
        """Небольшая сетка на val для моделей из ДЗ (по умолчанию выключена: берутся значения из ДЗ)."""
        sp, cfg = self.splits, self.cfg
        with self.stage("3a. подбор гиперпараметров моделей"):
            ev = FastEvaluator(sp.R_train, sp.val_df, self._val_users(2000))
            def ndcg(model):
                return ev.evaluate(np.asarray(model.score(ev.users)))["nDCG@10"]
            pop = {m: ndcg(PopularityRecommender("mean_rating", m).fit(sp.R_train)) for m in (1, 10, 25, 40, 55)}
            ok = [m for m, v in pop.items() if v >= 0.95 * max(pop.values())]
            cfg.params["popularity"] = {"strategy": "mean_rating", "min_ratings": int(max(ok))}
            cf = {(k, s): ndcg(ItemBasedCF(k=k, shrinkage=s).fit(sp.R_train)) for k, s in itertools.product((5, 10, 20, 50), (0, 10))}
            k, s = max(cf, key=cf.get)
            cfg.params["item_cf"] = {"k": int(k), "shrinkage": float(s)}
            n_max = min(800, min(self.ds.shape) - 2)
            pure = {f: ndcg(PureSVDRecommender(n_factors=f).fit(sp.R_train)) for f in sorted({200, 500, n_max})}
            cfg.params["pure_svd"] = {"n_factors": int(max(pure, key=pure.get))}
            self.tuning_tables = {"popularity": pop, "item_cf": cf, "pure_svd": pure}
        return self

    def _val_users(self, n: int, segment: int | None = None) -> np.ndarray:
        sp = self.splits
        rel = np.unique(sp.val_df.loc[sp.val_df["rating"] >= LIKE_THRESHOLD, "u"])
        if segment is not None:
            rel = rel[self.store_train.user_segment[rel] == segment]
        return self.rng.choice(rel, min(n, len(rel)), replace=False)

    def fit_train_models(self) -> "Pipeline":
        """Модели на train: нужны только для подбора весов гибрида и баланса на val."""
        sp, cfg = self.splits, self.cfg
        with self.stage("3b. обучение моделей на train"):
            ev = None
            if cfg.use_neural:
                ev = FastEvaluator(sp.R_train, sp.val_df, self._val_users(3000))
            self.models_train = build_components(cfg, self.ds, sp.train_df, sp.R_train, self.store_train, evaluator=ev)
            if cfg.use_neural:
                self.neural_best_epoch = self.models_train["neural"].best_epoch_
                self.neural_history = pd.DataFrame(self.models_train["neural"].history_)
        return self

    # ------------------------------------------------------------------ 4. гибрид
    def tune_hybrid(self) -> "Pipeline":
        """Подбор весов для четырёх вариантов гибрида (по сегментам и общий) на val."""
        sp, cfg = self.splits, self.cfg
        names_hw = ["pop", "svd", "pure", "cf", "content"]
        names_feat = names_hw + ["meta", "prior"]
        names_nn = names_feat + (["neural"] if cfg.use_neural else [])
        names_hw = [n for n in names_hw if n in self.models_train]
        names_feat = [n for n in names_feat if n in self.models_train]
        names_nn = [n for n in names_nn if n in self.models_train]
        counts = np.asarray(sp.R_train.getnnz(axis=0)).ravel()
        conf = item_confidence(counts, cfg.conf_c)
        seg_users = {s: self._val_users(cfg.tune_users_per_segment, s) for s in range(3)}
        val_rel = np.unique(sp.val_df.loc[sp.val_df["rating"] >= LIKE_THRESHOLD, "u"])
        pools = {s: int((self.store_train.user_segment[val_rel] == s).sum()) for s in range(3)}
        total = sum(pools.values())

        with self.stage("4. подбор весов гибрида (val)"):
            z, ev = {}, {}
            for s, users in seg_users.items():
                if len(users) == 0:
                    continue
                z[s] = {k: normalize_scores(self.models_train[k].score(users)) for k in names_nn}
                ev[s] = FastEvaluator(sp.R_train, sp.val_df, users)
            share = {s: pools[s] / total for s in z}            # естественные доли сегментов среди пользователей val
            self.segment_share = share

            self.searches = {}

            def opt(segments, names):
                parts = [(z[s], ev[s], share[s]) for s in segments]
                res = optimize_weights(parts, conf, names, n_random=cfg.search_random,
                                       n_sweeps=cfg.search_sweeps, seed=cfg.seed)
                self.searches[(tuple(segments), tuple(names))] = res      # история поиска (для графиков)
                return res

            def with_fallback(by_seg: dict, fallback: dict) -> dict:
                return {s: by_seg.get(s, fallback) for s in range(3)}      # сегмент без пользователей val -> общие веса

            self.weights = {"hw": {None: opt(list(z), names_hw).weights},
                            "feat": {None: opt(list(z), names_feat).weights}}
            self.weights["feat_seg"] = with_fallback({s: opt([s], names_feat).weights for s in z},
                                                     self.weights["feat"][None])
            if cfg.use_neural:
                self.weights["nn"] = {None: opt(list(z), names_nn).weights}
                self.weights["nn_seg"] = with_fallback({s: opt([s], names_nn).weights for s in z},
                                                       self.weights["nn"][None])
            # одиночные модели на тех же пользователях val (для сравнения)
            self.val_single = pd.DataFrame({
                SEGMENT_TITLES[s]: {k: ev[s].evaluate(z[s][k])["nDCG@10"] for k in names_nn} for s in z})
            self.val_single["все (взвешенно)"] = sum(share[s] * self.val_single[SEGMENT_TITLES[s]] for s in z)
            self.conf_train = conf
        return self

    def make_hybrid(self, models: dict, variant: str, counts: np.ndarray, store: FeatureStore) -> HybridRecommender:
        return HybridRecommender(models, self.weights[variant], counts, store.user_segment, c=self.cfg.conf_c)

    # ------------------------------------------------------------------ 5. баланс
    def tune_balance(self) -> "Pipeline":
        """Выбор параметра MMR и доли популярного в выдаче по val. Правило: самое сильное разнообразие (малое lambda),
        при котором nDCG@10 падает не более чем на mmr_tolerance от значения без переранжирования."""
        sp, cfg = self.splits, self.cfg
        variant = "nn_seg" if cfg.use_neural else "feat_seg"
        with self.stage("5. подбор баланса релевантность / разнообразие (val)"):
            users = np.sort(np.concatenate([self._val_users(1000, s) for s in range(3)]))
            counts = np.asarray(sp.R_train.getnnz(axis=0)).ravel()
            hyb = self.make_hybrid(self.models_train, variant, counts, self.store_train)
            pop_scores = self._pop_scores(self.models_train)
            rows = []
            for lam in cfg.mmr_grid:
                bal = BalanceConfig(self._sources(hyb), cfg.pool_popular, lam, 50)
                sys_ = HybridSystem(hyb, self.content.item_emb, pop_scores, bal).set_history(sp.R_train)
                r = evaluate_ranking(sys_, sp.R_train, sp.val_df, ks=(10,), users=users, item_emb=self.content.item_emb)
                rows.append({"lambda": lam, **r.summary[["nDCG@10", "R@10", "ild@10", "coverage@10", "novelty@10"]].to_dict()})
            self.balance_table = pd.DataFrame(rows)
            base = self.balance_table.loc[self.balance_table["lambda"] == 1.0, "nDCG@10"].iloc[0]
            ok = self.balance_table[self.balance_table["nDCG@10"] >= (1 - cfg.mmr_tolerance) * base]
            self.mmr_lambda = float(ok["lambda"].min())
            # доля слотов под популярное - отдельно по сегментам
            shares, share_rows = {}, []
            for s in range(3):
                seg_u = users[self.store_train.user_segment[users] == s]
                if len(seg_u) == 0:
                    shares[s] = 0.0
                    continue
                best_v, best_s = -1, 0.0
                for sh in cfg.pop_share_grid:
                    bal = BalanceConfig(self._sources(hyb), cfg.pool_popular, self.mmr_lambda, 50, {s: sh})
                    sys_ = HybridSystem(hyb, self.content.item_emb, pop_scores, bal).set_history(sp.R_train)
                    v = evaluate_ranking(sys_, sp.R_train, sp.val_df, ks=(10,), users=seg_u).summary["nDCG@10"]
                    share_rows.append({"сегмент": SEGMENT_TITLES[s], "доля популярного": sh, "nDCG@10": v})
                    if v > best_v * 1.005:
                        best_v, best_s = v, sh
                shares[s] = best_s
            self.pop_share_table = pd.DataFrame(share_rows)
            self.pop_share = shares
        return self

    def _sources(self, hyb: HybridRecommender) -> dict:
        return {k: n for k, n in self.cfg.pool_sources.items() if k in hyb.models}

    def _pop_scores(self, models) -> np.ndarray:
        """Популярность для слотов «популярное» и пользователей без истории: число лайков в train."""
        return np.asarray(models["content"].likes_.sum(axis=0)).ravel()

    # ------------------------------------------------------------------ 6. финальные модели
    def fit_final(self) -> "Pipeline":
        sp, cfg = self.splits, self.cfg
        with self.stage("6. финальное обучение на train+val"):
            ncfg = replace(cfg.neural, epochs=self.neural_best_epoch) if cfg.use_neural else None
            self.models = build_components(cfg, self.ds, sp.trainval_df, sp.R_trainval, self.store_tv, neural_cfg=ncfg)
            counts = np.asarray(sp.R_trainval.getnnz(axis=0)).ravel()
            self.hybrids = {v: self.make_hybrid(self.models, v, counts, self.store_tv) for v in self.weights}
            final_variant = "nn_seg" if cfg.use_neural else "feat_seg"
            hyb = self.hybrids[final_variant]
            bal = BalanceConfig(self._sources(hyb), cfg.pool_popular, self.mmr_lambda, 50, self.pop_share)
            self.system = HybridSystem(hyb, self.content.item_emb, self._pop_scores(self.models), bal)
            self.system.set_history(sp.R_trainval)
        return self

    # ------------------------------------------------------------------ 7. оценка
    def evaluate(self) -> "Pipeline":
        sp, cfg = self.splits, self.cfg
        with self.stage("7. оценка на test"):
            emb = self.content.item_emb
            self.results = {}
            for k, m in self.models.items():
                self.results[MODEL_TITLES[k]] = evaluate_ranking(m, sp.R_trainval, sp.test_df, ks=cfg.ks, item_emb=emb)
            titles = {"hw": "Hybrid-1: модели ДЗ, общие веса", "feat": "Hybrid-2: + признаки (автор/серия, приор)",
                      "feat_seg": "Hybrid-3: + веса по сегментам", "nn": "Hybrid-4a: + Two-Tower, общие веса",
                      "nn_seg": "Hybrid-4: + Two-Tower, веса по сегментам"}
            for v, h in self.hybrids.items():
                self.results[titles[v]] = evaluate_ranking(h, sp.R_trainval, sp.test_df, ks=cfg.ks, item_emb=emb)
            self.results["Hybrid system (пул + баланс)"] = evaluate_ranking(self.system, sp.R_trainval, sp.test_df,
                                                                          ks=cfg.ks, item_emb=emb)
            self.hybrid_titles = titles
        return self

    def metrics_by_segment(self, metric: str = "nDCG@10") -> pd.DataFrame:
        seg = self.store_tv.user_segment
        out = {}
        for name, r in self.results.items():
            pu = r.per_user
            out[name] = pu[metric].groupby(seg[pu.index]).mean().rename(index=SEGMENT_TITLES)
        return pd.DataFrame(out).T

    # ------------------------------------------------------------------ 8. выдача и экспорт
    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Рекомендации итоговой системы для пользователя по его id из ratings.csv."""
        u = self.ds.user_index(user_id)
        rec = self.system.rank(np.array([u]), n, self.splits.R_trainval[[u]])[0]
        out = self.ds.books.loc[rec, ["book_id", "title", "authors"]].reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, n + 1))
        return out

    def export(self) -> "Pipeline":
        cfg, sp = self.cfg, self.splits
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with self.stage("8. экспорт результатов"):
            users = np.sort(self.rng.choice(self.ds.n_users, min(cfg.n_export_users, self.ds.n_users), replace=False))
            rec = self.system.rank(users, cfg.n_recs, sp.R_trainval[users])
            books = self.ds.books
            pd.DataFrame({
                "user_id": np.repeat(self.ds.user_ids[users], cfg.n_recs), "rank": np.tile(np.arange(1, cfg.n_recs + 1), len(users)),
                "book_id": books["book_id"].to_numpy()[rec.ravel()], "title": books["title"].to_numpy()[rec.ravel()],
                "authors": books["authors"].to_numpy()[rec.ravel()],
            }).to_csv(out / "recommendations.csv", index=False)
            cols = [f"{m}@{k}" for k in cfg.ks for m in ("P", "R", "nDCG")] + ["coverage@10", "novelty@10", "ild@10"]
            pd.DataFrame({n: r.summary for n, r in self.results.items()}).T[cols].to_csv(out / "metrics_test.csv")
            self.metrics_by_segment().to_csv(out / "metrics_by_segment_ndcg10.csv")
            (out / "hybrid_weights.json").write_text(json.dumps(
                {v: {str(s): w for s, w in ws.items()} for v, ws in self.weights.items()}, indent=1, default=float))
            (out / "config.json").write_text(json.dumps(cfg.to_json(), indent=1, ensure_ascii=False))
            (out / "report.json").write_text(json.dumps({
                "timings_sec": self.timings, "mmr_lambda": self.mmr_lambda, "pop_share": {str(k): v for k, v in self.pop_share.items()},
                "neural_best_epoch": getattr(self, "neural_best_epoch", None),
                "n_users": self.ds.n_users, "n_items": self.ds.n_items, "n_ratings": len(self.ds.ratings)}, indent=1, default=float))
        return self

    def run(self) -> "Pipeline":
        self.load().prepare()
        if self.cfg.tune_models:
            self.tune_hyperparams()
        return self.fit_train_models().tune_hybrid().tune_balance().fit_final().evaluate().export()
