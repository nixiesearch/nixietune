from typing import Any
import torch
from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple


@dataclass
class ROCAUC:
    def __call__(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor = torch.empty) -> Any:
        return roc_auc_score(target, preds)


@dataclass
class Histogram:
    buckets: int = field(default=30)
    range: Optional[Tuple[float, float]] = field(default=(0.0, 1.0))

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor = torch.empty) -> Any:
        positives = []
        negatives = []
        for score, label in zip(preds.tolist(), target.tolist()):
            if label == 0:
                negatives.append(score)
            else:
                positives.append(score)
        pos_hist, _ = np.histogram(positives, bins=self.buckets, range=self.range)
        neg_hist, _ = np.histogram(negatives, bins=self.buckets, range=self.range)
        pos_sum = len(positives)
        neg_sum = len(negatives)
        pos_hist_norm = [round(c / pos_sum, 5) for c in pos_hist]
        neg_hist_norm = [round(c / neg_sum, 5) for c in neg_hist]
        return {"pos": pos_hist_norm, "neg": neg_hist_norm}
