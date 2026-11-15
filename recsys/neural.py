"""Продвинутая часть: нейросетевая Two-Tower модель (двухбашенная) для совместных эмбеддингов.

Идея. Пользователь и книга кодируются двумя независимыми нейросетями в общее пространство размерности d.
Релевантность - косинусная близость векторов, поэтому для всех книг скоры пользователя считаются одним
матричным произведением, а эмбеддинги книг можно посчитать заранее и положить в ANN-индекс.

* Башня книги: ID-эмбеддинг (с dropout, чтобы башня работала и по одному контенту) + текстовый эмбеддинг
  книги (SVD от TF-IDF тегов и названия) + числовые признаки книги и её тематический профиль -> MLP.
* Башня пользователя: усреднённые эмбеддинги истории лайков (ID- и текстовые) + числовые признаки
  пользователя (средний рейтинг, активность, широта вкуса и т.д.) -> MLP. Нового пользователя достаточно
  описать его лайками, переобучать модель не нужно.
* Обучение: пары (пользователь, понравившаяся книга). Целевая книга вырезается из истории пользователя
  (иначе сеть просто «подсмотрит» ответ), часть истории случайно выбрасывается (history dropout).
  Лосс - softmax по книгам других пар батча (in-batch negatives) с logQ-коррекцией частоты книг, чтобы
  популярные книги не доминировали в негативах.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from .features import FeatureStore
from .metrics import Recommender

USER_FEATURES = ["log_n_ratings", "mean_rating", "rating_std", "like_share", "dislike_share",
                 "mainstream", "genre_entropy", "series_share", "author_repeat"]
ITEM_FEATURES = ["log_n_ratings", "bayes_mean", "rating_std", "rating_entropy", "like_ratio",
                 "log_gr_count", "gr_avg_rating", "gr_rating_entropy", "gr_like_share", "year"]


@dataclass
class TwoTowerConfig:
    emb_dim: int = 64
    hidden: int = 256
    out_dim: int = 64
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-6
    temperature: float = 0.07
    hist_len: int = 60          # сколько лайков истории учитывает башня пользователя
    hist_dropout: float = 0.3   # доля истории, случайно скрываемая при обучении
    id_dropout: float = 0.3     # вероятность занулить ID-эмбеддинг книги (учит башню работать по контенту)
    use_id: bool = True
    use_content: bool = True
    use_features: bool = True
    pool: str = "mean"          # пулинг истории: "mean" или "meanmax" (среднее + максимум, сохраняет резкие сигналы)
    seed: int = 42


def _standardize(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)).astype(np.float32)


class _TwoTowerNet(nn.Module):
    def __init__(self, n_items: int, content: np.ndarray, user_feat: np.ndarray, item_feat: np.ndarray,
                 cfg: TwoTowerConfig):
        super().__init__()
        self.cfg, self.n_items = cfg, n_items
        d, c = cfg.emb_dim, content.shape[1]
        self.register_buffer("content", torch.from_numpy(np.vstack([content, np.zeros((1, c), np.float32)])))
        self.register_buffer("user_feat", torch.from_numpy(user_feat))
        self.register_buffer("item_feat", torch.from_numpy(item_feat))
        self.hist_emb = nn.Embedding(n_items + 1, d, padding_idx=n_items)
        self.id_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.hist_emb.weight, std=0.1)
        nn.init.normal_(self.id_emb.weight, std=0.1)
        with torch.no_grad():
            self.hist_emb.weight[n_items].zero_()
        mult = 2 if cfg.pool == "meanmax" else 1
        u_in = (mult * d if cfg.use_id else 0) + (mult * c if cfg.use_content else 0) + (user_feat.shape[1] if cfg.use_features else 0)
        i_in = (d if cfg.use_id else 0) + (c if cfg.use_content else 0) + (item_feat.shape[1] if cfg.use_features else 0)
        assert u_in > 0 and i_in > 0, "включите хотя бы один источник информации"
        mlp = lambda n_in: nn.Sequential(nn.Linear(n_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.out_dim))
        self.user_mlp, self.item_mlp = mlp(u_in), mlp(i_in)
        self.register_buffer("id_trained", torch.ones(n_items))        # 0 для книг без лайков в train

    def user_vec(self, users: torch.Tensor, hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        parts = []
        def pool(x: torch.Tensor) -> list[torch.Tensor]:
            out = [(x * mask.unsqueeze(-1)).sum(dim=1) / cnt]
            if self.cfg.pool == "meanmax":
                out.append(x.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values * (mask.sum(1, keepdim=True) > 0))
            return out
        if self.cfg.use_id:
            parts += pool(self.hist_emb(hist))
        if self.cfg.use_content:
            parts += pool(self.content[hist])
        if self.cfg.use_features:
            parts.append(self.user_feat[users])
        return F.normalize(self.user_mlp(torch.cat(parts, dim=1)), dim=-1)

    def item_vec(self, items: torch.Tensor, id_drop: float = 0.0) -> torch.Tensor:
        parts = []
        if self.cfg.use_id:
            e = self.id_emb(items) * self.id_trained[items].unsqueeze(-1)
            if id_drop > 0:
                e = e * (torch.rand(len(items), 1, device=e.device) > id_drop).float()
            parts.append(e)
        if self.cfg.use_content:
            parts.append(self.content[items])
        if self.cfg.use_features:
            parts.append(self.item_feat[items])
        return F.normalize(self.item_mlp(torch.cat(parts, dim=1)), dim=-1)


class TwoTowerRecommender(Recommender):
    """Two-Tower рекомендатель: score(u, i) = cos(user_tower(u), item_tower(i))."""

    def __init__(self, cfg: TwoTowerConfig | None = None, like_threshold: float = 4):
        self.cfg = cfg or TwoTowerConfig()
        self.like_threshold = like_threshold
        self.name = "Two-Tower"

    # ------------------------------------------------------------------ обучение
    def _hist_matrix(self, likes: sparse.csr_matrix, rng: np.random.Generator) -> np.ndarray:
        n_users, n_items = likes.shape
        h = np.full((n_users, self.cfg.hist_len), n_items, dtype=np.int64)
        for u in range(n_users):
            items = likes.indices[likes.indptr[u]:likes.indptr[u + 1]]
            if len(items) > self.cfg.hist_len:
                items = rng.choice(items, self.cfg.hist_len, replace=False)
            h[u, :len(items)] = items
        return h

    def fit(self, train: sparse.csr_matrix, store: FeatureStore, evaluator=None, verbose: bool = False):
        """train - матрица оценок (пользователь x книга); store - признаки, посчитанные по тем же данным.

        evaluator (FastEvaluator, необязательный) - после каждой эпохи считается nDCG@10 на его пользователях;
        в конце восстанавливаются веса лучшей эпохи. История обучения - в self.history_.
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        n_users, n_items = train.shape
        likes = (train >= self.like_threshold).astype(np.float32).tocsr()
        likes.eliminate_zeros()
        self.likes_ = likes
        self.hist_ = torch.from_numpy(self._hist_matrix(likes, rng))

        uf = _standardize(store.user_features[USER_FEATURES].to_numpy(np.float64))
        itf = np.hstack([store.item_features[ITEM_FEATURES].to_numpy(np.float64),
                         store.content.genre.astype(np.float64)])
        self.net = _TwoTowerNet(n_items, store.content.item_emb, uf, _standardize(itf), cfg)
        coo = likes.tocoo()
        pos_u, pos_i = torch.from_numpy(coo.row.astype(np.int64)), torch.from_numpy(coo.col.astype(np.int64))
        freq = np.bincount(coo.col, minlength=n_items).astype(np.float64)
        self.net.id_trained.copy_(torch.from_numpy((freq > 0).astype(np.float32)))
        log_q = torch.from_numpy(np.log((freq + 1.0) / (freq.sum() + n_items)).astype(np.float32))

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        n_pos, bs = len(pos_u), cfg.batch_size
        self.history_, best, best_state = [], -1.0, None
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(n_pos)
            total = 0.0
            for s in range(0, n_pos - bs + 1, bs):
                idx = perm[s:s + bs]
                u, i = pos_u[idx], pos_i[idx]
                hist = self.hist_[u]
                mask = ((hist != i[:, None]) & (hist != n_items) & (torch.rand(hist.shape) > cfg.hist_dropout)).float()
                uv = self.net.user_vec(u, hist, mask)
                iv = self.net.item_vec(i, cfg.id_dropout)
                logits = uv @ iv.T / cfg.temperature - log_q[i][None, :]
                same = (i[:, None] == i[None, :]) & ~torch.eye(len(i), dtype=torch.bool)
                logits = logits.masked_fill(same, -1e9)
                loss = F.cross_entropy(logits, torch.arange(len(i)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            sched.step()
            rec = {"epoch": epoch + 1, "loss": total / max(n_pos // bs, 1)}
            if evaluator is not None:
                self._refresh()
                rec["val_nDCG@10"] = evaluator.evaluate(self.score(evaluator.users))["nDCG@10"]
                if rec["val_nDCG@10"] > best:
                    best, best_state = rec["val_nDCG@10"], copy.deepcopy(self.net.state_dict())
            self.history_.append(rec)
            if verbose:
                print(rec)
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_epoch_ = int(np.argmax([r["val_nDCG@10"] for r in self.history_]) + 1)
        self._refresh()
        return self

    # ------------------------------------------------------------------ инференс
    @torch.no_grad()
    def _refresh(self):
        """Пересчитывает эмбеддинги всех книг (после обучения они фиксированы)."""
        self.net.eval()
        n = self.net.n_items
        self.item_vecs_ = torch.cat([self.net.item_vec(torch.arange(s, min(s + 4096, n)))
                                     for s in range(0, n, 4096)])

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray) -> torch.Tensor:
        u = torch.as_tensor(np.asarray(users), dtype=torch.long)
        hist = self.hist_[u]
        mask = (hist != self.net.n_items).float()
        return self.net.user_vec(u, hist, mask)

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        users = np.asarray(users)
        out = [(self.user_vectors(users[s:s + 4096]) @ self.item_vecs_.T).numpy() for s in range(0, len(users), 4096)]
        return np.vstack(out)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
