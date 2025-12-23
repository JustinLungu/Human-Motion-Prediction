"""Very small smoke tests for models package."""

from pathlib import Path
import sys

# Ensure `src` is on sys.path so `models` and `dataloader` import work when running
# this file directly.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.run_models import run_models


def test_baseline_smoke():
    results = run_models()
    print(results.keys())

if __name__ == "__main__":
    test_baseline_smoke()
    print("test_baseline_smoke passed")
