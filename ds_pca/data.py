"""Read portable condition tables and prepared FEKO intensity arrays."""
import numpy as np
import pandas as pd

from ._core import KEY


def read_conditions(path):
    frame = pd.read_csv(path, dtype={"case_id": str}).reset_index(drop=True)
    required = ["case_id"] + KEY
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Conditions must be nonempty and contain no missing values.")
    if frame.case_id.duplicated().any():
        raise ValueError("Each case_id must be unique.")
    if not np.isfinite(frame[KEY[:3]].to_numpy(dtype=float)).all():
        raise ValueError("Physical conditions must be finite numbers.")
    if (frame.wind_speed_mps < 0).any() or (frame.frequency_GHz <= 0).any():
        raise ValueError("Wind speed must be nonnegative and frequency positive.")
    if not frame.theta_i_deg.between(0, 90).all():
        raise ValueError("Incidence zenith angles must lie between 0 and 90 degrees.")
    if not frame.polarization.isin(["TE", "TM"]).all():
        raise ValueError("Polarization must be TE or TM.")
    return frame


def load_training_data(conditions_path, maps_path):
    conditions = read_conditions(conditions_path)
    keys = {"case_ids", "intensity_linear", "theta_deg", "phi_deg", "solid_angle_sr"}
    with np.load(maps_path, allow_pickle=False) as arrays:
        if not keys.issubset(arrays.files):
            raise ValueError(f"Missing NPZ keys: {sorted(keys - set(arrays.files))}")
        case_ids = arrays["case_ids"].astype(str)
        maps = arrays["intensity_linear"]
        theta = arrays["theta_deg"]
        phi = arrays["phi_deg"]
        weights = arrays["solid_angle_sr"]
    if case_ids.ndim != 1 or len(set(case_ids)) != len(case_ids):
        raise ValueError("NPZ case_ids must be a unique one-dimensional array.")
    if theta.ndim != 1 or phi.ndim != 1 or len(theta) < 2 or len(phi) < 2:
        raise ValueError("theta_deg and phi_deg must be one-dimensional grids.")
    if not np.isfinite(theta).all() or not np.isfinite(phi).all():
        raise ValueError("Direction grids must be finite.")
    if not (np.diff(theta) > 0).all() or not (np.diff(phi) > 0).all():
        raise ValueError("Direction grids must be strictly increasing.")
    if not (np.isclose(theta[0], 0) and np.isclose(theta[-1], 90)):
        raise ValueError("The zenith grid must cover 0 to 90 degrees.")
    if phi[0] < 0 or phi[-1] >= 360:
        raise ValueError("Use unique azimuth samples in [0, 360); omit duplicate 360.")
    if maps.shape != (len(case_ids), len(theta), len(phi)):
        raise ValueError("Intensity shape must be (case, zenith, azimuth).")
    if weights.shape != (len(theta), len(phi)):
        raise ValueError("Solid-angle weights must match the direction grid.")
    if not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Intensity must be finite, real and nonnegative.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Every solid-angle weight must be finite and positive.")
    if not np.isclose(weights.sum(), 2 * np.pi):
        raise ValueError("Upper-hemisphere weights must sum to 2*pi steradians.")
    lookup = {case_id: i for i, case_id in enumerate(case_ids)}
    absent = sorted(set(conditions.case_id) - set(lookup))
    if absent:
        raise ValueError(f"CSV cases absent from NPZ: {absent[:5]}")
    indices = [lookup[case_id] for case_id in conditions.case_id]
    return conditions, maps[indices], theta, phi, weights
