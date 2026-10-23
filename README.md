# Forecast vs Generation

A probabilistic forecast of German wind power generation, built from the ECMWF IFS ensemble (ENS) and verified against actual output.

```
ECMWF open data (ENS, 50 perturbed members, 100 m u/v, 0.25°)
  └─ ecmwf-opendata ─▶ GRIB2 ─▶ eccodes decode + crop to DE ─▶ cube (member, step, lat, lon)
        └─ |u,v| ─▶ shear to hub height ─▶ power curve ─▶ capacity factor per cell
              └─ capacity-weighted regions (onshore land / offshore sea) ─▶ national MW per member
                    └─ quantiles P10…P90  ◀── compared with ──▶  Energy-Charts / SMARD actuals
                          └─ FastAPI: /forecast /actuals /verify /chart (fan chart PNG)
```

![Fan chart: ECMWF ENS 2026-10-20 00Z forecast vs actual German wind generation](docs/fan_chart_2026-10-20T00.png)

## Results (4 ENS runs, 20–21 Oct 2026, 48 h each)

Each run was scored with scale factors fitted on the other three runs (leave-one-run-out):

| Run | MAE raw → calibrated (GW) | CRPS raw → calibrated (GW) | P10–P90 coverage raw → calibrated |
|---|---|---|---|
| 2026-10-20 00Z | 2.98 → 2.01 | 2.12 → 1.47 | 59% → 76% |
| 2026-10-20 12Z | 6.55 → 2.39 | 4.89 → 1.58 | 18% → 94% |
| 2026-10-21 00Z | 4.90 → 2.07 | 3.50 → 1.53 | 35% → 88% |
| 2026-10-21 12Z | 5.69 → 2.58 | 3.89 → 1.76 | 53% → 100% |
| **Mean** | **5.03 → 2.26** | **3.60 → 1.58** | **41% → 90%** |

The raw ensemble catches the timing of the ramps but converts wind to too much power. The scale factors are 0.85 onshore and 0.74 offshore. After calibration the P10–P90 band is slightly too wide (90% coverage against a target of about 80%), so spread calibration (EMOS) is the next step.

| Raw | Calibrated |
|---|---|
| ![raw](docs/fan_chart_2026-10-20T12_raw.png) | ![calibrated](docs/fan_chart_2026-10-20T12_calibrated.png) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

On Windows, `eccodes` installs from a binary wheel. If that fails, use `conda install -c conda-forge eccodes`.

## Run

```bash
# a run from 2-3 days ago, 48 h horizon, so actuals already exist for most of it
.venv/Scripts/python scripts/run_pipeline.py --run 2026-10-20T00 --max-step 48

# older runs: the ECMWF server keeps about 4 days; Azure has the archive (slower)
.venv/Scripts/python scripts/run_pipeline.py --run 2026-10-10T00 --max-step 48 --source azure

# full 6-day horizon of the newest run
.venv/Scripts/python scripts/run_pipeline.py --run latest --max-step 144

# after several runs: fit per-fleet scale factors with leave-one-run-out scores
.venv/Scripts/python scripts/calibrate.py

# API + chart
.venv/Scripts/python -m uvicorn windfc.api:app --reload
```

Then open http://127.0.0.1:8000/ for the chart or http://127.0.0.1:8000/docs for the API.

**Download size:** every field is global (about 1.3 MB), even though only Germany is kept. A 48 h run (50 members × 2 params × 17 steps) is about 2.2 GB, and a 144 h run is about 6.4 GB. The client fetches only the needed fields through byte ranges from the `.index` files. The pipeline crops right after decoding, stores a ~20 MB cube in `data/cube/`, and deletes the raw GRIB files unless you pass `--keep-raw`. `--step-interval 6` halves the download.

**Sources:** use `ecmwf` (the default, fastest) for the last ~4 days and `azure` for older runs. AWS has the same archive, but S3 throttles the client's multi-range requests with `503 Slow Down`.

| Endpoint | What it returns |
|---|---|
| `GET /runs` | Cached runs and the installed capacity |
| `GET /forecast?run=latest&fleet=total&calibrated=false` | Ensemble mean and P10/P25/P50/P75/P90 per valid time (MW) |
| `GET /actuals?start=...&end=...` | Actual onshore, offshore and total wind output (MW, 15 min) |
| `GET /verify?run=latest` | MAE, RMSE and bias of the median, CRPS and P10–P90 coverage, both overall and by lead time |
| `GET /chart?run=latest` | Fan chart PNG with the actuals drawn over it |

`fleet` can be `total`, `onshore` or `offshore`.

## Code map

| File | Role |
|---|---|
| [config.yaml](config.yaml) | Crop box, power curves, availability, regions and installed capacity |
| [src/windfc/download.py](src/windfc/download.py) | `ecmwf-opendata` retrieval in step chunks, plus the land-sea mask |
| [src/windfc/decode.py](src/windfc/decode.py) | `eccodes` message loop, grid geometry and crop |
| [src/windfc/power.py](src/windfc/power.py) | Wind speed, shear and power curve |
| [src/windfc/aggregate.py](src/windfc/aggregate.py) | Regional cos(lat)-weighted CF × capacity, giving MW per member; quantiles |
| [src/windfc/actuals.py](src/windfc/actuals.py) | Energy-Charts `public_power` with SMARD `chart_data` as fallback; alignment to valid times |
| [src/windfc/verify.py](src/windfc/verify.py) | CRPS (ensemble form), MAE, RMSE, bias, coverage, by lead time |
| [src/windfc/calibrate.py](src/windfc/calibrate.py) | Least-squares scale factor per fleet |
| [src/windfc/api.py](src/windfc/api.py), [chart.py](src/windfc/chart.py) | FastAPI service and matplotlib fan chart |

## Method notes (interview talking points)

- **Why an ensemble:** the 50 perturbed members (the open data does not include the control) are equally likely, and they sample the uncertainty in the initial conditions and the model. The spread between them gives a usable forecast distribution, and traders price risk from those quantiles, not from a single deterministic line. ENS runs 4 times a day (00/06/12/18 UTC). The 00/12 runs go out to 15 days; the 06/18 runs go out to 6 days.
- **Wind at 100 m:** this is close to modern hub heights (about 100–160 m). A power-law shear (α = 0.14) lifts the wind to 120 m for onshore turbines. Real shear depends on atmospheric stability: it is larger at night and in stable conditions, and that is a known source of error.
- **Power curve:** a generic fleet curve per fleet (cut-in 3 m/s, rated about 12–14 m/s, cut-out 25 m/s). It is applied per grid cell before averaging, because the curve is non-linear. Averaging the wind first and then applying the curve would bias the result.
- **Capacity weighting:** about 67 GW onshore is split into rough regional boxes, and the north and east carry the most weight. Offshore has about 8.3 GW in the North Sea and 1.6 GW in the Baltic. With the land-sea mask, onshore regions use only land cells and offshore regions use only sea cells.
- **Verification:**
  - MAE and RMSE of the median show deterministic skill.
  - CRPS = E|X − y| − ½E|X − X′| scores the whole distribution. It is in MW, and with a single member it reduces to MAE.
  - P10–P90 coverage should be about 80%. Lower means the ensemble is under-dispersive, which is typical for raw ENS in the short range.
  - Scores are broken down by lead time to show how skill decays.
- **Calibration:** one multiplicative scale per fleet is fitted on other runs (leave-one-run-out), so it corrects capacity, curve and availability bias without in-sample cheating.

## Limitations and next steps

- Capacities and region boxes are rough. Next step: per-turbine capacity and hub height from the Marktstammdatenregister.
- There is no modelling of curtailment (redispatch, negative prices), icing, wakes or outages. Availability is a single constant factor.
- The fleet power curve is generic. A better approach is to learn an empirical curve (wind → CF) from history.
- Calibration only rescales. Next step: EMOS / NGR or quantile regression to also fix the spread, and a comparison against ICON-EU and the DWD ICON-D2-EPS ensemble.
- The forecast is instantaneous every 3 h, while the actuals are 15-min values averaged over an hour around each valid time.
