"""Check nonlinear derivatives and portable fitted-model inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ds_pca import DSPCAModel
from ds_pca.data import load_training_data
from ds_pca.models import extended_value_jac, fit_extended, log_coordinates


class ModelChecks(unittest.TestCase):
    def test_analytic_jacobian(self):
        rng = np.random.default_rng(20260920)
        u, v = rng.uniform(-1.5, 1.5, (2, 80))
        p = np.array([20, np.log(12), np.log(17), np.log(9), np.log(.8), np.log(1.2), .17])
        _, analytic = extended_value_jac(p, u, v)
        numerical = np.empty_like(analytic)
        for col in range(len(p)):
            plus, minus = p.copy(), p.copy()
            plus[col] += 1e-6
            minus[col] -= 1e-6
            numerical[:, col] = (extended_value_jac(plus, u, v, False)
                                 - extended_value_jac(minus, u, v, False)) / 2e-6
        self.assertLess(np.max(abs(analytic - numerical) / (1 + abs(numerical))), 1e-6)

    def test_fit_and_saved_model(self):
        theta = np.arange(0., 91., 5.)
        phi = np.arange(0., 360., 10.)
        weights = np.broadcast_to(
            (np.cos(np.deg2rad(np.maximum(theta - 2.5, 0)))
             - np.cos(np.deg2rad(np.minimum(theta + 2.5, 90))))[:, None] * np.deg2rad(10),
            (len(theta), len(phi))).copy()
        p = np.array([20, np.log(12), np.log(17), np.log(9), np.log(.8), np.log(1.2), .17])
        u, v = log_coordinates(theta, phi, 50)
        target = extended_value_jac(p, u, v, False).reshape(weights.shape)
        fitted, _ = fit_extended(target, 50, theta, phi, weights)
        recovered = extended_value_jac(fitted, u, v, False).reshape(weights.shape)
        self.assertLess(np.sqrt(np.sum((target-recovered)**2 * weights) / weights.sum()), .02)
        rng = np.random.default_rng(17)
        maps = 10 ** ((target + rng.normal(0, .3, (8, *weights.shape))) / 10)
        conditions = pd.DataFrame({"wind_speed_mps": [6.] * 8, "frequency_GHz": [4.] * 8,
                                   "theta_i_deg": [50.] * 8, "polarization": ["TE"] * 8})
        query = conditions.iloc[:2].copy()
        query.loc[1, "theta_i_deg"] = 35.
        with tempfile.TemporaryDirectory() as temp:
            model = DSPCAModel(theta, phi, weights).fit(conditions, maps)
            before = model.predict(query)
            path = Path(temp) / "ds_pca.npz"
            model.save(path)
            loaded = DSPCAModel.load(path)
            np.testing.assert_array_equal(before, loaded.predict(query))
            np.testing.assert_allclose(before, loaded.predict(query.iloc[::-1])[::-1], rtol=1e-10)
            self.assertTrue(np.isfinite(before).all() and (before > 0).all())

    def test_training_data_aligns_by_case_identifier(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            frame = pd.DataFrame({"case_id": ["b", "a"], "wind_speed_mps": [6., 6.],
                                  "frequency_GHz": [4., 4.], "theta_i_deg": [20., 50.],
                                  "polarization": ["TE", "TM"]})
            frame.to_csv(temp / "conditions.csv", index=False)
            maps = np.stack([np.ones((2, 2)), np.full((2, 2), 2.)])
            np.savez(temp / "maps.npz", case_ids=np.array(["a", "b"]), intensity_linear=maps,
                     theta_deg=np.array([0., 90.]), phi_deg=np.array([0., 180.]),
                     solid_angle_sr=np.full((2, 2), np.pi / 2))
            _, loaded, *_ = load_training_data(temp / "conditions.csv", temp / "maps.npz")
            np.testing.assert_array_equal(loaded, maps[::-1])
            frame.loc[0, "case_id"] = "missing"
            frame.to_csv(temp / "conditions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "absent from NPZ"):
                load_training_data(temp / "conditions.csv", temp / "maps.npz")


if __name__ == "__main__":
    unittest.main()
