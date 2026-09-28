"""Fit a model to explicitly selected training conditions."""
import argparse
from pathlib import Path

from ds_pca import DSPCAModel
from ds_pca.data import load_training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    conditions, maps, theta, phi, weights = load_training_data(args.conditions, args.maps)
    if len(conditions) < 4:
        parser.error("Provide at least four training samples for the default K=3 use case.")
    model = DSPCAModel(theta, phi, weights).fit(conditions, maps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved DS+PCA (K=3, ridge=0.001), fitted to {len(conditions)} samples: {args.output}")


if __name__ == "__main__":
    main()
