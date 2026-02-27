# train_ltr.py
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.search_engine.ltr.model import (
    TinyLTRModel,
    DEFAULT_FEATURE_ORDER,
)


@dataclass
class Batch:
    x: torch.Tensor        # (B, L, F)
    y: torch.Tensor        # (B, L)
    mask: torch.Tensor     # (B, L) bool
    qids: List[int]


# -----------------------------
# Data loading
# -----------------------------
class JsonlLTRDataset(Dataset):
    """
    One line == one query impression:
      {"qid":..., "query":..., "docs":[{"doc_id":..., "label":..., "features":{...}}, ...]}
    """

    def __init__(self, path: Path, feature_order: List[str], max_docs: int | None = None):
        self.path = Path(path)
        self.feature_order = feature_order
        self.max_docs = max_docs

        self.rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        qid = int(row["qid"])
        docs = row["docs"]

        if self.max_docs is not None:
            docs = docs[: self.max_docs]

        feats = []
        labels = []
        for d in docs:
            f = d["features"]
            feats.append([float(f.get(name, 0.0)) for name in self.feature_order])
            labels.append(float(d.get("label", 0.0)))

        x = torch.tensor(feats, dtype=torch.float32)     # (L, F)
        y = torch.tensor(labels, dtype=torch.float32)    # (L,)
        return qid, x, y


def collate_fn(batch_items: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Batch:
    # pad to max L in batch
    qids = [qid for (qid, _, _) in batch_items]
    lengths = [x.shape[0] for (_, x, _) in batch_items]
    max_l = max(lengths)
    fdim = batch_items[0][1].shape[1]

    xs = torch.zeros((len(batch_items), max_l, fdim), dtype=torch.float32)
    ys = torch.zeros((len(batch_items), max_l), dtype=torch.float32)
    mask = torch.zeros((len(batch_items), max_l), dtype=torch.bool)

    for i, (_, x, y) in enumerate(batch_items):
        l = x.shape[0]
        xs[i, :l, :] = x
        ys[i, :l] = y
        mask[i, :l] = True

    return Batch(x=xs, y=ys, mask=mask, qids=qids)


# -----------------------------
# Metrics: MRR@10, nDCG@10
# -----------------------------
@torch.no_grad()
def mrr_at_k(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, k: int = 10) -> float:
    # scores/labels/mask: (B, L)
    B, L = scores.shape
    total = 0.0
    for b in range(B):
        valid = mask[b]
        s = scores[b][valid]
        y = labels[b][valid]
        if s.numel() == 0:
            continue

        # sort desc
        order = torch.argsort(s, descending=True)
        y_sorted = y[order]
        topk = y_sorted[: min(k, y_sorted.numel())]

        # first relevant
        rr = 0.0
        for i in range(topk.numel()):
            if topk[i].item() > 0:
                rr = 1.0 / (i + 1)
                break
        total += rr
    return total / B


@torch.no_grad()
def ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, k: int = 10) -> float:
    # Binary labels are fine; formula matches slides. :contentReference[oaicite:6]{index=6}
    B, L = scores.shape
    total = 0.0
    for b in range(B):
        valid = mask[b]
        s = scores[b][valid]
        y = labels[b][valid]
        if s.numel() == 0:
            continue

        order = torch.argsort(s, descending=True)
        y_sorted = y[order]
        topk = y_sorted[: min(k, y_sorted.numel())]

        # DCG
        dcg = 0.0
        for i in range(topk.numel()):
            rel = topk[i].item()
            if rel > 0:
                dcg += rel / math.log2(1.0 + (i + 1) + 0.0)  # log2(1+rank), rank is 1-based

        # IDCG
        y_ideal = torch.sort(y, descending=True).values
        ideal_topk = y_ideal[: min(k, y_ideal.numel())]
        idcg = 0.0
        for i in range(ideal_topk.numel()):
            rel = ideal_topk[i].item()
            if rel > 0:
                idcg += rel / math.log2(1.0 + (i + 1) + 0.0)

        total += (dcg / idcg) if idcg > 0 else 0.0

    return total / B


# -----------------------------
# Loss: Listwise Softmax Loss
# -----------------------------
def listwise_softmax_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Implements:
      L = - sum_i y_i * log softmax(s)_i
    For multiple positives: normalize y to sum 1 over valid docs.
    This corresponds to the listwise softmax loss shown in the slides. :contentReference[oaicite:8]{index=8}
    """
    # scores already masked with -1e9 at pads, but keep mask for label normalization
    y = labels.clone()
    y = y.masked_fill(~mask, 0.0)

    # normalize labels per query (avoid all-zero)
    denom = y.sum(dim=1, keepdim=True).clamp_min(1.0)
    y = y / denom

    logp = F.log_softmax(scores, dim=1)
    loss = -(y * logp).sum(dim=1).mean()
    return loss


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model: TinyLTRModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mrrs, ndcgs, losses = [], [], []
    for batch in loader:
        x = batch.x.to(device)
        y = batch.y.to(device)
        mask = batch.mask.to(device)

        scores = model(x, mask)
        loss = listwise_softmax_loss(scores, y, mask)

        losses.append(loss.item())
        mrrs.append(mrr_at_k(scores, y, mask, k=10))
        ndcgs.append(ndcg_at_k(scores, y, mask, k=10))

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "mrr@10": float(sum(mrrs) / max(1, len(mrrs))),
        "ndcg@10": float(sum(ndcgs) / max(1, len(ndcgs))),
    }


def flatten_train_features(train_ds: JsonlLTRDataset) -> torch.Tensor:
    all_feats = []
    for _, x, _ in tqdm(train_ds, desc="Collect train features"):
        all_feats.append(x)  # (L, F)
    return torch.cat(all_feats, dim=0)  # (N, F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)

    ap.add_argument("--max_docs", type=int, default=20, help="truncate per query (see slides about truncation)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    _script_dir = Path(__file__).resolve().parent
    ap.add_argument("--logdir", type=str, default=str(_script_dir / "output" / "runs"))
    ap.add_argument("--out", type=str, default=str(_script_dir / "output" / "ltr_model.pt"))

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    feature_order = DEFAULT_FEATURE_ORDER
    num_features = len(feature_order)

    train_ds = JsonlLTRDataset(Path(args.train), feature_order, max_docs=args.max_docs)
    val_ds = JsonlLTRDataset(Path(args.val), feature_order, max_docs=args.max_docs)
    test_ds = JsonlLTRDataset(Path(args.test), feature_order, max_docs=args.max_docs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)

    model = TinyLTRModel(num_features=num_features, hidden=args.hidden, dropout=args.dropout).to(device)

    # Fit normalization on TRAIN set and store inside model buffers (saved with state_dict)
    x_all = flatten_train_features(train_ds)  # (N, F)
    model.norm.fit(x_all)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=args.logdir)

    global_step = 0
    best_val = -1.0
    best_path = Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch.x.to(device)
            y = batch.y.to(device)
            mask = batch.mask.to(device)

            scores = model(x, mask)
            loss = listwise_softmax_loss(scores, y, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            opt.step()

            # track loss by iteration (as demanded)
            writer.add_scalar("train/loss", loss.item(), global_step)

            if global_step % 200 == 0:
                # quick train metrics on this batch (optional but useful)
                with torch.no_grad():
                    mrr = mrr_at_k(scores, y, mask, k=10)
                    ndcg = ndcg_at_k(scores, y, mask, k=10)
                writer.add_scalar("train/mrr@10_batch", mrr, global_step)
                writer.add_scalar("train/ndcg@10_batch", ndcg, global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # full validation at epoch end (don’t look at test until the end) :contentReference[oaicite:10]{index=10}
        val_metrics = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/mrr@10", val_metrics["mrr@10"], epoch)
        writer.add_scalar("val/ndcg@10", val_metrics["ndcg@10"], epoch)

        # choose best model by ndcg@10 (common) :contentReference[oaicite:11]{index=11}
        if val_metrics["ndcg@10"] > best_val:
            best_val = val_metrics["ndcg@10"]
            payload = {
                "state_dict": model.state_dict(),
                "feature_order": feature_order,
                "max_docs": args.max_docs,
                "hidden": args.hidden,
                "dropout": args.dropout,
            }
            torch.save(payload, best_path)

        print(f"[epoch {epoch}] val: {val_metrics}")

    # Final test evaluation ONCE (after you’re done tuning) :contentReference[oaicite:12]{index=12}
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"[FINAL] test: {test_metrics}")

    writer.close()
    print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()