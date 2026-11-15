"""Тесты: метрики, отсутствие утечек и сквозной запуск пайплайна на синтетических данных
(это же демонстрация перезапуска процесса на новых данных)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from recsys.data import build_dataset, load_raw, make_splits                                # noqa: E402
from recsys.hybrid import mmr_select                                                         # noqa: E402
from recsys.metrics import Recommender, evaluate_ranking, ranking_metrics                    # noqa: E402
from recsys.neural import TwoTowerConfig                                                     # noqa: E402
from recsys.pipeline import Pipeline, PipelineConfig, synthetic_dataset                      # noqa: E402


def test_ranking_metrics_hand_computed():
    m = ranking_metrics(np.array([[1, 0, 1]], dtype=bool), np.array([4]), ks=(1, 3))
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert np.isclose(m["P@3"][0], 2 / 3) and np.isclose(m["R@3"][0], 2 / 4)
    assert np.isclose(m["nDCG@3"][0], dcg / idcg) and np.isclose(m["nDCG@1"][0], 1.0)


def test_evaluate_ranking_excludes_seen_items():
    class Fixed(Recommender):
        def __init__(self, s):
            self.s = s

        def score(self, users):
            return self.s[users]

    train = sparse.csr_matrix(np.array([[5, 0, 0, 0, 0]], dtype=np.float32))
    test = pd.DataFrame({"u": [0, 0], "i": [1, 2], "rating": [5, 3]})
    res = evaluate_ranking(Fixed(np.array([[100, 9, 5, 4, 3]], dtype=np.float32)), train, test, ks=(1, 2))
    assert res.summary["P@1"] == 1.0 and res.summary["nDCG@2"] == 1.0


def test_mmr_lambda_one_is_plain_ranking():
    rng = np.random.default_rng(0)
    cand = np.tile(np.arange(20), (3, 1))
    rel = -np.arange(20, dtype=np.float32)[None, :].repeat(3, 0)
    emb = rng.normal(size=(20, 4)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert (mmr_select(cand, rel, emb, 5, 1.0) == cand[:, :5]).all()
    div = mmr_select(cand, rel, emb, 5, 0.3)
    assert len(set(div[0])) == 5


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("books")
    synthetic_dataset(d, n_users=300, n_books=150, seed=1)
    return d


def test_splits_have_no_leakage(synthetic):
    ds = build_dataset(load_raw(synthetic))
    sp = make_splits(ds)
    assert not set(zip(sp.trainval_df.u, sp.trainval_df.i)) & set(zip(sp.test_df.u, sp.test_df.i))
    assert len(sp.train_df) + len(sp.val_df) == len(sp.trainval_df)


def test_end_to_end_pipeline_on_new_data(synthetic, tmp_path):
    cfg = PipelineConfig(
        data_dir=str(synthetic), out_dir=str(tmp_path / "out"),
        params={"popularity": {"strategy": "mean_rating", "min_ratings": 3}, "item_cf": {"k": 5, "shrinkage": 0.0},
                "svd": {"n_factors": 5, "n_epochs": 5, "reg_all": 0.1, "random_state": 0},
                "pure_svd": {"n_factors": 20}},
        neural=TwoTowerConfig(epochs=3, batch_size=256, hidden=32, emb_dim=16, out_dim=16, hist_len=20),
        n_topics=5, emb_dim=16, tune_users_per_segment=60, search_random=3,
        search_sweeps=1, n_export_users=50)
    pipe = Pipeline(cfg, verbose=False).run()
    out = Path(cfg.out_dir)
    for f in ("recommendations.csv", "metrics_test.csv", "hybrid_weights.json", "config.json", "report.json"):
        assert (out / f).exists(), f
    metrics = pd.read_csv(out / "metrics_test.csv", index_col=0)
    assert np.isfinite(metrics.to_numpy()).all()
    assert metrics.loc["Hybrid system (пул + баланс)", "nDCG@10"] > metrics.loc["Random (reference)", "nDCG@10"]
    # в выдаче нет уже прочитанных книг
    recs = pd.read_csv(out / "recommendations.csv")
    pairs = set(zip(pipe.splits.trainval_df["user_id"], pipe.splits.trainval_df["book_id"]))
    assert not any((u, b) in pairs for u, b in zip(recs["user_id"], recs["book_id"]))
    assert len(pipe.recommend(int(pipe.ds.user_ids[0]), 5)) == 5
    json.loads((out / "report.json").read_text())
