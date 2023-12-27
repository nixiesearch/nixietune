from typing import List, Dict
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from torchmetrics.retrieval.average_precision import RetrievalMAP
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR
import torch
import logging
from transformers.trainer_utils import EvalPrediction
from sentence_transformers import util

logger = logging.getLogger()


class EvalMetrics:
    def __init__(self, metrics: List[str]) -> None:
        self.logger = logging.getLogger()
        self.metrics = metrics
        logger.info(f"Eval metrics: {metrics}")

    def compute(self, embeds: EvalPrediction) -> Dict[str, float]:
        query_embeds = torch.from_numpy(embeds.predictions[0]["sentence_embedding"]).cpu()
        doc_embeds = torch.from_numpy(embeds.predictions[1]["sentence_embedding"]).cpu()
        scores = util.pairwise_cos_sim(query_embeds, doc_embeds)
        _, indexes = torch.unique_consecutive(query_embeds, return_inverse=True, dim=0)
        targets = torch.from_numpy(embeds.label_ids)
        result = {}
        for metric_name in self.metrics:
            match metric_name.lower().split("@"):
                case ["ndcg"]:
                    metric = RetrievalNormalizedDCG()
                    result[metric_name] = metric(preds=scores, target=targets, indexes=indexes)
                case ["ndcg", k]:
                    metric = RetrievalNormalizedDCG(top_k=int(k))
                    result[metric_name] = metric(preds=scores, target=targets, indexes=indexes)
                case ["map"]:
                    result[metric_name] = RetrievalMAP()
                case ["map", k]:
                    result[metric_name] = RetrievalMAP(top_k=int(k))
                case ["mrr"]:
                    result[metric_name] = RetrievalMRR()
                case ["mrr", k]:
                    result[metric_name] = RetrievalMRR(top_k=int(k))
                case other:
                    logger.warn(f"Metric type {other} is not yet supported")

        return result
