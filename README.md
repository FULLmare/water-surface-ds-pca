# DS+PCA for water-surface scattering intensity

Final DS+PCA implementation and fitted model associated with the manuscript
*An extended directive model for water-surface scattering intensity*.

**Repository:** <https://github.com/FULLmare/water-surface-ds-pca>

The model predicts scattering intensity from wind speed, carrier frequency,
incidence zenith angle and polarisation. Its extended directive scattering (DS)
envelope describes the dominant water-surface lobe. A mean residual and three
solid-angle-weighted principal components describe the remaining intensity
variation in decibels. Both parameter and score regression use a Gaussian kernel
with ridge coefficient 0.001.

This release contains one final model: **DS+PCA, K=3, ridge=0.001**.

## Run the fitted model

Python 3.10 or later, NumPy and pandas are required. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python predict.py --output outputs/prediction.npz
```

This reads `weights/ds_pca_final.npz` and the two example conditions in
`examples/query_conditions.csv`. No fitting or FEKO installation is required.
Use a different condition table with:

```bash
python predict.py --conditions my_conditions.csv --output outputs/my_prediction.npz
```

Each CSV row needs:

| Column | Meaning |
| --- | --- |
| `case_id` | Unique query or training-sample identifier |
| `wind_speed_mps` | Wind speed in metres per second |
| `frequency_GHz` | Carrier frequency in GHz |
| `theta_i_deg` | Incidence zenith angle in degrees |
| `polarization` | `TE` or `TM` |

The output NPZ contains `case_ids`, `intensity_linear`, `theta_deg`, `phi_deg`
and `solid_angle_sr`. Intensity has shape `(number of queries, 91, 360)` for the
provided checkpoint. Angles are in degrees; solid-angle weights are in steradians.
Decibel intensity is `10*log10(max(intensity_linear, 1e-8))`.

For Python use:

```python
from ds_pca import DSPCAModel
from ds_pca.data import read_conditions

model = DSPCAModel.load("weights/ds_pca_final.npz")
conditions = read_conditions("examples/query_conditions.csv")
intensity = model.predict(conditions)
```

## Fitted model and provenance

The released checkpoint is the final DS+PCA model fitted to all **137 R1-R3
training samples**, spanning **46 conditions and nine water-surface realisations**.
It corresponds to the manuscript's **known-condition setting**. R4-R5 samples
were not used to fit these weights. The separate interpolation settings in the
paper used their own fits with the respective condition value withheld; their
checkpoints are not part of this single-model release.

| Setting | Value |
| --- | --- |
| Residual PCA components | 3 |
| Gaussian-kernel ridge coefficient | 0.001 |
| Extended DS envelope parameters | 7 |
| Training wind speeds | 3, 6, 9 m/s |
| Training frequencies | 2, 4, 6 GHz |
| Training incidence zenith angles | 20, 50, 70 degrees |
| Polarisations | TE, TM |
| Observation grid | 0-90 degrees zenith, 0-359 degrees azimuth |

The training conditions form a subset of the Cartesian grid. Inputs specify
physical conditions, so output is not tied to a particular random surface
realisation. Intensity follows the manuscript's normalisation by incident-field
amplitude squared and surface area; it is not an absolute received-power value.

The checkpoint was exported from the existing fitted model by retaining the
three selected PCA components and the regression coefficients for ridge 0.001.
The envelope parameters, residual mean and retained basis values were preserved;
no refitting was performed. Checkpoint hashes, grid dimensions, training counts
and the comparison against the original implementation are recorded in
`weights/model_metadata.json`.

## Fit the same DS+PCA model to prepared data

```bash
python train.py --conditions data/train_conditions.csv --maps data/train_maps.npz --output outputs/ds_pca.npz
python predict.py --model outputs/ds_pca.npz --conditions my_conditions.csv --output outputs/prediction.npz
```

The CSV must contain only the samples selected for fitting. The NPZ must contain:

| Array | Shape / content |
| --- | --- |
| `case_ids` | One-dimensional string array matching CSV sample identifiers |
| `intensity_linear` | `(samples, zenith samples, azimuth samples)` |
| `theta_deg` | Strictly increasing grid covering 0 to 90 degrees |
| `phi_deg` | Strictly increasing grid in `[0, 360)` |
| `solid_angle_sr` | Positive weights for each grid cell, summing to `2*pi` |

The loader aligns arrays by `case_id` and allows the CSV to select a subset.
Prepared FEKO data must omit the duplicate 360-degree azimuth and average linear
intensities across azimuth at the zenith. The input is consistently normalised
linear intensity; raw electromagnetic-field files are not parsed by this code.
Keep fitting and evaluation samples separate when constructing the CSV.

## Software checks

```bash
python -m unittest discover -s tests -v
```

The tests check the envelope Jacobian, nonlinear fitting, saved-model prediction,
query ordering and CSV/array alignment. A small synthetic fitting example is also
available:

```bash
python examples/create_demo.py
python train.py --conditions demo_data/train_conditions.csv --maps demo_data/train_maps.npz --output outputs/demo_model.npz
python predict.py --model outputs/demo_model.npz --conditions demo_data/query.csv --output outputs/demo_prediction.npz
```

These synthetic examples test the software workflow; they are not FEKO data or
lake measurements. Raw simulation data and measurement records remain separate
from this model release. Processed figure and table data accompany the manuscript.
