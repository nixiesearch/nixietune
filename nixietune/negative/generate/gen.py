from nixietune.negative.generate.args import GenArgs
from nixietune.format.json import JSONDataset
from sentence_transformers import SentenceTransformer


class NegativeGenerator:
    def __init__(self, args: GenArgs) -> None:
        self.args = args
        self.queries = JSONDataset.load(path=args.query_file, split="train").select_columns(["query"])
        self.corpus = JSONDataset.load(path=args.doc_file, split="train").select_columns(["doc"])
        self.model = SentenceTransformer(args.model_name)
