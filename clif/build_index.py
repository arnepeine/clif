"""Entry point: python -m clif.build_index ...  (same as `clif build-index ...`)"""

import sys

from clif.cli import main

if __name__ == "__main__":
    sys.exit(main(["build-index", *sys.argv[1:]]))
