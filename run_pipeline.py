"""Запуск сквозного пайплайна рекомендательной системы одной командой.

    python run_pipeline.py --data-dir data --out-dir artifacts/run1

Для новых данных достаточно указать другую папку с четырьмя CSV (ratings, books, tags, book_tags) в формате
Goodbooks-10k. Все результаты (метрики, рекомендации, веса гибрида, конфигурация) сохраняются в --out-dir.
"""
import argparse
import dataclasses
import warnings

from recsys.neural import TwoTowerConfig
from recsys.pipeline import Pipeline, PipelineConfig

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-neural", action="store_true", help="не обучать Two-Tower (только классический гибрид)")
    ap.add_argument("--neural-epochs", type=int, default=None)
    ap.add_argument("--tune-models", action="store_true", help="подобрать гиперпараметры моделей на val (дольше)")
    ap.add_argument("--users-per-segment", type=int, default=None, help="размер выборки val для подбора весов")
    args = ap.parse_args()

    cfg = PipelineConfig(data_dir=args.data_dir, out_dir=args.out_dir, seed=args.seed,
                         use_neural=not args.no_neural, tune_models=args.tune_models)
    if args.neural_epochs:
        cfg.neural = dataclasses.replace(cfg.neural, epochs=args.neural_epochs)
    if args.users_per_segment:
        cfg.tune_users_per_segment = args.users_per_segment
    pipe = Pipeline(cfg).run()
    print("\nГотово. Метрики на test (K=10):")
    cols = ["P@10", "R@10", "nDCG@10", "ild@10"]
    print(__import__("pandas").DataFrame({n: r.summary for n, r in pipe.results.items()}).T[cols].round(4).to_string())
    print(f"\nФайлы с результатами: {cfg.out_dir}/")


if __name__ == "__main__":
    main()
