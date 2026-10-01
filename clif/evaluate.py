"""Entry point: python -m clif.evaluate ...  (same as `clif evaluate ...`)"""

import sys

from clif.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
