"""Create small synthetic intensity data for testing the command-line workflow.

These are analytical demonstration data, not FEKO simulations or measurements.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo_data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    theta = np.arange(0.0, 91.0, 5.0)
    phi = np.arange(0.0, 360.0, 10.0)
    lower = np.maximum(theta - 2.5, 0)
    upper = np.minimum(theta + 2.5, 90)
    weights = np.broadcast_to((np.cos(np.deg2rad(lower)) - np.cos(np.deg2rad(upper)))[:, None]
                              * np.deg2rad(10), (len(theta), len(phi))).copy()
    conditions = []
    maps = []
    rng = np.random.default_rng(20260928)
    for incidence in [20, 50, 70]:
        for realisation in range(4):
            case_id = f"demo_{incidence}_{realisation}"
            conditions.append(dict(case_id=case_id, wind_speed_mps=6.0, frequency_GHz=4.0,
                                   theta_i_deg=float(incidence), polarization="TE"))
            t, p, ti = np.deg2rad(theta[:, None]), np.deg2rad(phi[None, :]), np.deg2rad(incidence)
            separation = np.arccos(np.clip(np.cos(t)*np.cos(ti) + np.sin(t)*np.sin(ti)*np.cos(p), -1, 1))
            intensity_db = 8 - 18*separation**1.2 + rng.normal(0, 0.8, weights.shape)
            intensity = 10**(intensity_db / 10)
            intensity[0, :] = intensity[0, :].mean()
            maps.append(intensity)
    frame = pd.DataFrame(conditions)
    frame.to_csv(args.output / "train_conditions.csv", index=False)
    np.savez_compressed(args.output / "train_maps.npz", case_ids=frame.case_id.to_numpy(dtype=str),
                        intensity_linear=np.stack(maps), theta_deg=theta, phi_deg=phi, solid_angle_sr=weights)
    pd.DataFrame([dict(case_id="query_35", wind_speed_mps=6.0, frequency_GHz=4.0,
                       theta_i_deg=35.0, polarization="TE")]).to_csv(args.output / "query.csv", index=False)
    print(f"Created 12 synthetic training samples and one query in {args.output}")


if __name__ == "__main__":
    main()
