from argparse import ArgumentParser
from typing import Dict, List, Optional
from tqdm import tqdm
import json
from dataclasses import dataclass
import csv
import random
from collections import defaultdict


@dataclass
class QRel:
    query: str
    document: str
    rel: int


@dataclass
class NixieDoc:
    query: Optional[str]
    doc: str
    neg: Optional[List[str]]

    def to_json(self) -> Dict[str, str]:
        result = {"doc": self.doc}
        if self.query:
            result["query"] = self.query
        if self.neg:
            result["neg"] = self.neg
        return result


def load_json(path: str) -> Dict[str, str]:
    dict = {}
    with open(path, "r") as f:
        for line in tqdm(f, desc=f"loading ${path}"):
            item = json.loads(line)
            if "title" in item and len(item["title"]) > 0:
                dict[item["_id"]] = item["title"] + " " + item["text"]
            else:
                dict[item["_id"]] = item["text"]
    return dict


def load_tsv(path: str, queries: Dict[str, str], corpus: Dict[str, str]) -> List[QRel]:
    qrels = []
    missing_query = 0
    missing_doc = 0
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"loading qrels from ${path}"):
            if row["query-id"] not in queries:
                missing_query += 1
            elif row["corpus-id"] not in corpus:
                missing_doc += 1
            else:
                qrels.append(QRel(row["query-id"], row["corpus-id"], int(row["score"])))
    print(
        f"Loaded {len(qrels)} qrels, dropped {missing_doc+missing_query} docs (query={missing_query}, doc={missing_doc})"
    )
    return qrels


def save(path: str, docs: List[NixieDoc], limit: int):
    with open(path, "w") as f:
        for doc in tqdm(docs[:limit], desc="saving json"):
            f.write(json.dumps(doc.to_json()) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(prog="trec2json", description="Convert TREC JSON+TSV to Nixietune format")
    parser.add_argument("--corpus", action="store", help="path to corpus.json", required=True)
    parser.add_argument("--queries", action="store", help="path to queries.json")
    parser.add_argument("--qrels", action="store", help="path to qrels.tsv")
    parser.add_argument("--limit", action="store", default=100000, help="Take first N docs")
    parser.add_argument("--out", action="store", default="out.json", help="output file name")
    parser.add_argument("--pos", nargs="+", help="positive labels", default=[1, 2, 3], type=int)
    parser.add_argument("--neg", nargs="+", help="negative labels", default=[0], type=int)
    parser.add_argument("--seed", action="store", default=42, type=int, help="random seed")

    args = parser.parse_args()
    print(args)
    random.seed(args.seed)

    corpus = load_json(args.corpus)
    if not args.queries:
        # pure corpus export
        docs: List[NixieDoc] = []
        for id, doc in tqdm(corpus.items(), desc="converting"):
            docs.append(NixieDoc(doc=doc, query=None, neg=None))
        random.shuffle(docs)
        save(args.out, docs, args.limit)
    elif args.queries and args.qrels:
        # corpus+query
        docs: List[NixieDoc] = []
        queries = load_json(args.queries)
        qrels = load_tsv(args.qrels, queries=queries, corpus=corpus)
        qgroups: Dict[str, List[QRel]] = defaultdict(list)
        for qrel in tqdm(qrels, desc="grouping per query"):
            group = qgroups[qrel.query]
            group.append(qrel)
            qgroups[qrel.query] = group
        for qid, gqrels in tqdm(qgroups.items(), desc="converting"):
            positives = [qrel for qrel in gqrels if qrel.rel in args.pos]
            negatives = [corpus[qrel.document] for qrel in gqrels if qrel.rel in args.neg]
            for pos in positives:
                neg = negatives if len(negatives) > 0 else None
                doc = NixieDoc(query=queries[pos.query], doc=corpus[pos.document], neg=neg)
                docs.append(doc)
        random.shuffle(docs)
        save(args.out, docs, args.limit)
    else:
        raise Exception("wrong arguments")

    print("done")
