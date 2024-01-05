from nixietune.main import main


def test_esci_mixed():
    main(["self", "tests/config/esci_mixed.json"])


def test_esci_infonce():
    main(["self", "tests/config/esci_infonce.json"])


def test_esci_infonce_no_negatives():
    main(["self", "tests/config/esci_infonce_nonegs.json"])


def test_esci_contrastive():
    main(["self", "tests/config/esci_contrastive.json"])


def test_esci_noeval():
    main(["self", "tests/config/esci_infonce_noeval.json"])
