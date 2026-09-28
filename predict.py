"""Predict scattering intensity from physical conditions and a saved model."""
import argparse
from pathlib import Path
import numpy as np

from ds_pca import DSPCAModel
from ds_pca.data import read_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--model", default=root / "weights/ds_pca_final.npz", type=Path)
    parser.add_argument("--conditions", default=root / "examples/query_conditions.csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    model = DSPCAModel.load(args.model)
    conditions = read_conditions(args.conditions)
    intensity = model.predict(conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, case_ids=conditions.case_id.to_numpy(dtype=str),
                        intensity_linear=intensity, theta_deg=model.theta, phi_deg=model.phi,
                        solid_angle_sr=model.w)
    print(f"Saved {len(conditions)} predictions with shape {intensity.shape}: {args.output}")


if __name__ == "__main__":
    main()
