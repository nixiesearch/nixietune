from nixietune.metrics.metrics import Histogram, ROCAUC
import torch


def test_histogram():
    scores = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    labels = torch.IntTensor([0, 0, 0, 1, 1, 1, 1, 1])
    metric = Histogram(3)
    result = metric(scores, labels)
    assert result == {"pos": [0.0, 0.0, 0.4], "neg": [0.0, 0.0, 1.0]}


def test_roc_auc():
    scores = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    labels = torch.IntTensor([0, 0, 0, 1, 1, 1, 1, 1])
    metric = ROCAUC()
    result = metric(scores, labels)
    assert result == 0.8
