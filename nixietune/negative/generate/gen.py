from nixietune.format.json import JSONDataset
from sentence_transformers import SentenceTransformer
import hnswlib
from typing import List
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

logger = logging.getLogger()


@dataclass
class Triplet:
    query: str
    doc: str
    neg: List[str] = field(default_factory=list)


class NegativeGenerator:
    def __init__(
        self, query_file: str, doc_file: str, model_name: str, query_prefix: str = "", doc_prefix: str = ""
    ) -> None:
        query = JSONDataset.load(path=query_file, split="train").to_dict()
        self.queries = []
        self.qpos = {}
        for q, d in tqdm(zip(query["query"], query["doc"]), desc="aggregating"):
            positives = self.qpos.get(q, [])
            positives.append(d)
            self.qpos[q] = positives
            self.queries.append(q)
        self.corpus = JSONDataset.load(path=doc_file, split="train").to_dict()["doc"]
        logger.info(f"Loaded queries={len(self.queries)} corpus={len(self.corpus)}")
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def generate(self, num_negs: int, batch_size: int = 128) -> List[Triplet]:
        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=len(self.corpus), ef_construction=200, M=16)
        logger.info("Encoding queries")
        encoded_queries = self.model.encode(
            sentences=[self.query_prefix + q for q in self.queries],
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Encoding docs")
        encoded_docs = self.model.encode(
            sentences=[self.doc_prefix + d for d in self.corpus],
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Building index")
        self.index.add_items(encoded_docs, list(range(len(encoded_docs))))
        self.index.set_ef(50)
        logger.info("Generating negatives")
        labels, distances = self.index.knn_query(encoded_queries, k=num_negs)

        result = []
        for qid, negs in enumerate(labels.tolist()):
            q = self.queries[qid]
            positives = self.qpos[q]
            neg_docs = [self.corpus[d] for d in negs]
            filtered_negs = [d for d in neg_docs if d not in positives]
            for p in positives:
                result.append(Triplet(q, p, filtered_negs))
        return result
