"""Entry point: python -m clif.import_rf2 ...  (same as `clif import-rf2 ...`)"""

import sys

from clif.cli import main

if __name__ == "__main__":
    sys.exit(main(["import-rf2", *sys.argv[1:]]))
