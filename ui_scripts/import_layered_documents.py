"""UI subprocess entry point for the reusable layered document converter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.layered_import import main

if __name__ == "__main__":
    raise SystemExit(main())
