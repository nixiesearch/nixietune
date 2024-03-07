from nixietune.negative.generate.gen import NegativeGenerator, Triplet


def test_gen():
    gen = NegativeGenerator(
        query_file="tests/negative/generate/data/query.jsonl",
        doc_file="tests/negative/generate/data/corpus.jsonl",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    results = gen.generate(2)
    assert results == [
        Triplet("hello", "hi", ["world", "daddy"]),
        Triplet("pasta", "spaghetti", ["sausage", "daddy"]),
    ]
