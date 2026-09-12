"""Strict public class import gate using real installed dependencies, without weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.model_registry import validate_model_imports

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arches', nargs='*')
    args = parser.parse_args()
    import extensions_built_in.diffusion_models
    failures = validate_model_imports(args.arches or None)
    print(json.dumps(failures, indent=2))
    raise SystemExit(bool(failures))
