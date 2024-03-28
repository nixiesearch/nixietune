from typing import List, Dict
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from torchmetrics.retrieval.average_precision import RetrievalMAP
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR
import torch
import logging
from transformers.trainer_utils import EvalPrediction
from sentence_transformers import util
from transformers import PreTrainedTokenizerBase
import numpy as np
from nixietune.metrics.metrics import ROCAUC, Histogram

logger = logging.getLogger()


class EvalMetrics:
    def __init__(self, metrics: List[str], tokenizer: PreTrainedTokenizerBase) -> None:
        self.logger = logging.getLogger()
        self.metrics = {}
        self.tokenizer = tokenizer
        for metric_name in metrics:
            match metric_name.lower().split("@"):
                case ["ndcg"]:
                    self.metrics[metric_name] = RetrievalNormalizedDCG()
                case ["ndcg", k]:
                    self.metrics[metric_name] = RetrievalNormalizedDCG(top_k=int(k))
                case ["map"]:
                    self.metrics[metric_name] = RetrievalMAP()
                case ["map", k]:
                    self.metrics[metric_name] = RetrievalMAP(top_k=int(k))
                case ["mrr"]:
                    self.metrics[metric_name] = RetrievalMRR()
                case ["mrr", k]:
                    self.metrics[metric_name] = RetrievalMRR(top_k=int(k))
                case ["auc"]:
                    self.metrics[metric_name] = ROCAUC()
                case ["hist"]:
                    self.metrics[metric_name] = Histogram(buckets=50, range=(0.0, 1.0))
                case other:
                    logger.warn(f"Metric type {other} is not yet supported")
        self.sep_token_id = tokenizer.sep_token_id
        logger.info(f"Eval metrics: {metrics}")

    def compute_ce(self, embeds: EvalPrediction) -> Dict[str, float]:
        # find positions of [sep] token for all query-doc pairs
        inputs = torch.from_numpy(embeds.inputs)

        sep_positions = torch.argmax((inputs == self.sep_token_id).to(dtype=torch.int), dim=-1)
        # mask everything beyond the [sep] token
        masked_queries = (torch.arange(inputs.size(1)) < sep_positions[..., None]) * inputs
        # char sum as a hashcode? why not!
        query_hashes = torch.sum(masked_queries, dim=1)

        _, indexes = torch.unique_consecutive(query_hashes, return_inverse=True, dim=0)
        scores = torch.from_numpy(embeds.predictions)
        targets = torch.from_numpy(embeds.label_ids).to(dtype=torch.int)
        # torch.set_printoptions(profile="full")
        # print(f"ind={indexes} scores={scores} target={targets}")
        result = {name: metric(preds=scores, target=targets, indexes=indexes) for name, metric in self.metrics.items()}
        return result

    def compute_bi(self, embeds: EvalPrediction) -> Dict[str, float]:
        query_embeds = torch.from_numpy(embeds.predictions[0]["sentence_embedding"]).cpu()
        doc_embeds = torch.from_numpy(embeds.predictions[1]["sentence_embedding"]).cpu()
        scores = util.pairwise_cos_sim(query_embeds, doc_embeds)
        _, indexes = torch.unique_consecutive(query_embeds, return_inverse=True, dim=0)
        targets = torch.from_numpy(embeds.label_ids)
        result = {name: metric(preds=scores, target=targets, indexes=indexes) for name, metric in self.metrics.items()}
        return result

    def compute_ranker(self, result: EvalPrediction) -> Dict[str, float]:
        preds = np.where(result.predictions != -100, result.predictions, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, device="cpu", skip_special_tokens=False)
        labels = np.where(result.label_ids != -100, result.label_ids, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, device="cpu", skip_special_tokens=False)
        print(f"labels: {decoded_preds}")
        print(f"preds: {decoded_labels}")
        return {}
