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
        for query, pos, negs, negscores in zip(batch["query"], batch["pos"], batch["neg"], batch["negscore"]):
            features.append([query, pos])
            labels.append(1.0)
            for neg, score in zip(negs, negscores):
                features.append([query, neg])
                labels.append(score)
        return {"features": features, "label": labels}


@dataclass
class QueryPosNegsLayout(Layout):
    num_negatives: int = 4

    def name(self) -> str:
        return "Unwrapping dataset to QPN layout"

    def schema(self) -> str:
        return Features({"features": [Sequence(Value("int32"))]})

    def unwrap(self, batch: Dict[str, Any]) -> Dict[str, List]:
        features = []
        for query, pos, negs in zip(batch["query"], batch["pos"], batch["neg"]):
            if self.num_negatives > 0 and len(negs) > 0:
                neg_sample = random.choices(negs, k=self.num_negatives)
                features.append([query, pos] + neg_sample)
            elif self.num_negatives == 0:
                features.append([query, pos])
        return {"features": features}
