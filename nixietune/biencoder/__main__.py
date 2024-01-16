import sys
from nixietune.log import setup_logging
from nixietune.biencoder.main import main

setup_logging()

if __name__ == "__main__":
    main(sys.argv)
