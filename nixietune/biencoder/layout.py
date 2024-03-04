from typing import Dict, List, Any
from dataclasses import dataclass
import random
from datasets import Features, Sequence, Value
import logging

logger = logging.getLogger()


class Layout:
    def schema(self) -> str:
        pass

    def desc(self) -> str:
        pass

    def unwrap(self, batch: Dict[str, Any]) -> Dict[str, List]:
        pass


@dataclass
class QueryDocLabelLayout(Layout):
    def name(self) -> str:
        return "Unwrapping dataset to QDL layout"

    def schema(self) -> str:
        return Features({"features": [Sequence(Value("int32"))], "label": Value("double")})

    def unwrap(self, batch: Dict[str, Any]) -> Dict[str, List]:
        features = []
        labels = []
        for query, docs, scores in zip(batch["query"], batch["docs"], batch["scores"]):
            for doc, score in zip(docs, scores):
                features.append([query, doc])
                labels.append(score)
        return {"features": features, "label": labels}


@dataclass
class QueryPosNegsLayout(Layout):
    positive_threshold: float = 0.5
    num_negatives: int = 4

    def name(self) -> str:
        return "Unwrapping dataset to QPN layout"

    def schema(self) -> str:
        return Features({"features": [Sequence(Value("int32"))]})

    def unwrap(self, batch: Dict[str, Any]) -> Dict[str, List]:
        features = []
        for query, docs, scores in zip(batch["query"], batch["docs"], batch["scores"]):
            positives = [doc for doc, score in zip(docs, scores) if score >= self.positive_threshold]
            negatives = [doc for doc, score in zip(docs, scores) if score < self.positive_threshold]
            if self.num_negatives > 0 and len(negatives) > 0:
                for pos in positives:
                    neg_sample = random.choices(negatives, k=self.num_negatives)
                    features.append([query, pos] + neg_sample)
            elif self.num_negatives == 0:
                for pos in positives:
                    features.append([query, pos])
        return {"features": features}
