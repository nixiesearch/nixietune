from nixietune.querygen.dataset import Prompter


def test_make_prompt():
    prompter = Prompter("mistralai/Mistral-7B-v0.3", 32)
    result = prompter.make_prompt({"doc": ["hello world"]})
    assert result["length"] == [4]
    assert result["prompt"] == [
        "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\nhello world\n\n### Response:\n"
    ]
    print(result)


def test_prompt_trim():
    prompter = Prompter("mistralai/Mistral-7B-v0.3", 6)
    result = prompter.make_prompt({"doc": ["test " * 8]})
    assert result["length"] == [6]
    assert result["prompt"] == [
        "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\ntest test test test test\n\n### Response:\n"
    ]


def test_extract_doc():
    prompter = Prompter("mistralai/Mistral-7B-v0.3", 32)
    result = prompter.parse_doc(
        "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\nhello world\n\n### Response:\n"
    )
    assert result == "hello world"


def test_extract_query():
    prompter = Prompter("mistralai/Mistral-7B-v0.3", 32)
    result = prompter.parse_response(
        "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\nhello world\n\n### Response:\nquery"
    )
    assert result == "query"
