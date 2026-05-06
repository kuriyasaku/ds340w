from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.paths import CHECKPOINTS_DIR, LOGS_DIR
from src.config.settings import CONFIG
from src.data.slide_registry import SlideRegistry
from src.data.wsi_dataset import SlideBagDataset
from src.models.attention_mil import AttentionMIL
from src.train.losses import attention_entropy_regularizer, slide_classification_loss, top_bottom_margin_loss
from src.utils.io_utils import write_json
from src.utils.logging_utils import get_logger


logger = get_logger("train_mil", LOGS_DIR / "train_mil.log")


class MILTrainer:
    def __init__(self):
        self.device = torch.device(CONFIG.runtime.device if torch.cuda.is_available() else "cpu")
        self.model = AttentionMIL(
            input_dim=CONFIG.mil.input_dim,
            hidden_dim=CONFIG.mil.hidden_dim,
            attention_dim=CONFIG.mil.attention_dim,
            num_classes=CONFIG.mil.num_classes,
            dropout=CONFIG.mil.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG.train.lr, weight_decay=CONFIG.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CONFIG.train.epochs)
        self.best_metric = -1.0
        self.history: List[Dict] = []

    def _make_loaders(self):
        registry = SlideRegistry()
        train_ds = SlideBagDataset(registry.get_split("train"))
        val_frame = registry.get_split("val") if "val" in registry.frame["split"].unique() else registry.get_split("train").sample(min(16, len(registry.get_split("train"))), random_state=42)
        test_frame = registry.get_split("test") if "test" in registry.frame["split"].unique() else val_frame.copy()
        val_ds = SlideBagDataset(val_frame)
        test_ds = SlideBagDataset(test_frame)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader

    def _move_batch(self, batch):
        feats = batch["features"][0].to(self.device)
        labels = batch["label"].to(self.device)
        return feats, labels

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        preds, probs, labels = [], [], []
        for batch in tqdm(loader, leave=False):
            feats, target = self._move_batch(batch)
            out = self.model(feats)
            loss_main = slide_classification_loss(out["logits"], target)
            loss_rank = top_bottom_margin_loss(out["instance_prob"], out["attention"], int(target.item()))
            loss_attn = attention_entropy_regularizer(out["attention"])
            loss = loss_main + 0.2 * loss_rank + 0.0005 * loss_attn
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.train.gradient_clip)
            self.optimizer.step()
            total_loss += float(loss.item())
            prob = out["prob"].detach().cpu().numpy()[0]
            pred = int(prob.argmax())
            preds.append(pred)
            probs.append(float(prob[1]))
            labels.append(int(target.item()))
        return self._summarize_epoch(total_loss, len(loader), preds, probs, labels)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, stage: str) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        preds, probs, labels = [], [], []
        for batch in tqdm(loader, leave=False):
            feats, target = self._move_batch(batch)
            out = self.model(feats)
            loss = slide_classification_loss(out["logits"], target)
            total_loss += float(loss.item())
            prob = out["prob"].detach().cpu().numpy()[0]
            pred = int(prob.argmax())
            preds.append(pred)
            probs.append(float(prob[1]))
            labels.append(int(target.item()))
        metrics = self._summarize_epoch(total_loss, len(loader), preds, probs, labels)
        metrics["stage"] = stage
        return metrics

    def _summarize_epoch(self, total_loss: float, n_steps: int, preds: List[int], probs: List[float], labels: List[int]) -> Dict[str, float]:
        metrics = {}
        metrics["loss"] = float(total_loss / max(1, n_steps))
        metrics["accuracy"] = float(accuracy_score(labels, preds)) if len(labels) else 0.0
        metrics["f1"] = float(f1_score(labels, preds, zero_division=0)) if len(labels) else 0.0
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else 0.0
        except Exception:
            metrics["roc_auc"] = 0.0
        metrics["positive_rate"] = float(np.mean(preds)) if len(preds) else 0.0
        metrics["avg_prob"] = float(np.mean(probs)) if len(probs) else 0.0
        return metrics

    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> Path:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINTS_DIR / f"mil_epoch_{epoch:03d}.pt"
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(CONFIG),
            "metrics": val_metrics,
        }
        torch.save(payload, path)
        best_path = CHECKPOINTS_DIR / "best_mil.pt"
        if val_metrics["roc_auc"] >= self.best_metric:
            self.best_metric = val_metrics["roc_auc"]
            torch.save(payload, best_path)
        return path

    def fit(self) -> Dict:
        train_loader, val_loader, test_loader = self._make_loaders()
        patience = 0
        for epoch in range(1, CONFIG.train.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, "val")
            self.scheduler.step()
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            self.history.append(record)
            logger.info(f"epoch={epoch} train={train_metrics} val={val_metrics}")
            self.save_checkpoint(epoch, val_metrics)
            if val_metrics["roc_auc"] >= self.best_metric:
                patience = 0
            else:
                patience += 1
            if patience >= CONFIG.train.early_stop_patience:
                logger.info("early stopping triggered")
                break
        test_metrics = self.evaluate(test_loader, "test")
        summary = {"history": self.history, "test": test_metrics}
        write_json(summary, CHECKPOINTS_DIR / "training_summary.json")
        return summary


def main():
    trainer = MILTrainer()
    trainer.fit()


if __name__ == "__main__":
    main()
