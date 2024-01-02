from mteb import MTEB
import torch
from sentence_transformers import SentenceTransformer


class E5Model:
    def __init__(self, name):
        self.model = SentenceTransformer(name)
        self.model.to("cuda:0")
        self.model.eval()

    def encode_queries(self, queries, batch_size=32, **kwargs):
        processed = [f"query: {q}" for q in queries]
        return self.encode_impl(processed, batch_size)

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        processed = [f"passage: {q}" for q in corpus]
        return self.encode_impl(processed, batch_size)

    @torch.no_grad()
    def encode_impl(self, input_texts, batch_size):
        return self.model.encode(input_texts, normalize_embeddings=True, batch_size=batch_size)


model = E5Model("/home/shutty/code/microtune/model")
evaluation = MTEB(tasks=["FiQA2018", "SciFact", "NFCorpus"])
results = evaluation.run(model, output_folder="results/res-cont-yolo", batch_size=256, verbosity=2)
