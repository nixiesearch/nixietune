from nixietune.biencoder.main import main
import pytest


@pytest.mark.slow
def test_esci_infonce():
    main(["self", "tests/biencoder/config/esci_infonce.json"])


@pytest.mark.slow
def test_esci_noeval():
    main(["self", "tests/biencoder/config/esci_infonce_noeval.json"])
