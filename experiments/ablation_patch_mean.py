"""Train and evaluate CLS or patch-mean Static WeightedSum fusion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from experiments.ablation_fusion import main
except ImportError:
    from ablation_fusion import main


if __name__ == "__main__":
    main()
