# Model Card: Occupancy Transformer

## Model Details

| Field | Value |
|---|---|
| Name | `occupancy-transformer-v1.2` |
| Type | Transformer encoder (classification) |
| Task | Multi-horizon occupancy prediction |
| Framework | PyTorch 2.x |
| Last updated | 2024-Q1 |
| Maintainer | Concordia CIISE Lab |

## Intended Use

**Primary use**: Predict whether a building zone will be occupied at 5, 15, and 30 minutes in the future, given a 2-hour window of multi-sensor readings.

**Intended users**: Smart building management systems, energy optimization controllers, HVAC scheduling software.

**Out-of-scope uses**: Tracking or identifying specific individuals. The model operates only on aggregate sensor signals — no camera or biometric data.

## Training Data

| Dataset | Zones | Duration | Notes |
|---|---|---|---|
| ASHRAE Occupancy Benchmark | 84 | 3 years | Public; diverse building types |
| Concordia CIISE Lab | 12 | 18 months | Thesis research data; internal |
| OpenBuildingData | 31 | 6 months | Public |

Pre-training used ASHRAE data. Domain adaptation fine-tuning used Concordia data with 20% of parameters frozen.

## Evaluation

Evaluated on a held-out test set (20% split, stratified by building type):

| Metric | Value |
|---|---|
| F1 (binary, 5-min horizon) | 0.913 |
| F1 (binary, 15-min horizon) | 0.891 |
| F1 (binary, 30-min horizon) | 0.867 |
| AUC-ROC | 0.961 |
| Calibration ECE | 0.032 |

## Limitations

- Performance degrades for zones with highly irregular schedules (e.g., event spaces)
- Model assumes sensor health — missing or drifted sensors degrade accuracy
- Requires at least 30 minutes of warm-up data for reliable predictions on a new zone
- Not validated for outdoor or semi-outdoor spaces

## Ethical Considerations

This model does not process personally identifiable information. CO₂, temperature, humidity, and motion data are aggregate environmental signals. No individual tracking is performed or possible from these inputs.

## Caveats

Occupancy predictions should be used as advisory inputs to HVAC and energy systems, not as sole actuators. Human override should always be possible.
