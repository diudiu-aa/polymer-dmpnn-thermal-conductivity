"""Compatibility entry point: fixed-input Table 3 training, dry-run by default.

The former --project-root private-project requirement has been removed. Use
--seed, --arm and --run --output NEW_DIRECTORY; see reproduce/README.md.
"""
from train_graph_models import main

if __name__ == '__main__':
    raise SystemExit(main())
