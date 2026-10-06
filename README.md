# A 2D physics-informed neural network for a Hall effect thruster discharge

Code for the reference solution of the P5 laboratory thruster at 300 V, 5.4 A, in which the
electric potential is solved from Poisson's equation written in charge-separation form,
`n_e = n_i(1 + eta)` with `|eta| <= 1e-4` bounded by the output parametrization.

## Layout

```
het_pinn_2d/
├── code/
│   ├── mainScript.py     driver: configuration, training loop, reporting
│   ├── model.py          network and output parametrizations
│   └── utils.py          residuals, losses, sampling, plotting, metrics
├── haas_gallimore_p5_5p4A.csv    probe data (not included; see below)
└── output/               created at run time
```

## Requirements

Python 3.10 or later, with `torch`, `numpy` and `matplotlib`. Training runs on CPU in float64.

```sh
pip install torch numpy matplotlib
```

## Data

The reference profiles are reconstructed analytically from the published figures of Haas and
Gallimore (2000); the original tabulated probe data are not public. Place the reconstruction at
`haas_gallimore_p5_5p4A.csv` beside the `code/` directory. The script also looks in `code/` and
in the current working directory. Without it the data terms are disabled and the run becomes
purely residual-driven.

## Running

```sh
cd code
SEED_OVERRIDE=7 python3 mainScript.py
```

`SEED_OVERRIDE` sets the weight initialization; the collocation sampling uses the `SEED`
constant in `mainScript.py` and is independent of it. The reference solution uses weight seed 7
and collocation seed 2601.

Training takes a few hours on a laptop CPU. Progress prints every 250 epochs and a monitor
block every 500.

### Environment overrides

| Variable | Effect |
|---|---|
| `SEED_OVERRIDE` | weight initialization seed |
| `EPOCHS_OVERRIDE` | number of Adam epochs, default 15000 |
| `LBFGS_STEPS_OVERRIDE` | L-BFGS steps after Adam, default 0 |
| `OUTPUT_OVERRIDE` | output directory |
| `CKPT_IN_OVERRIDE` | checkpoint to resume from |
| `CKPT_OUT_OVERRIDE` | checkpoint path to write |
| `CENTERLINE_DATA_R_MM` | restrict the data terms to one radial line, in mm from the inner wall |

## Configuration

The physical and numerical settings are at the top of `mainScript.py`:

- `ALPHA_B = 1/200`, the Bohm coefficient of the anomalous collision term `nu_B = alpha_B * omega_ce`,
  which enters the three electron momentum equations only
- `energy_alpha = 2.5`, the ionization-energy multiplier in the electron energy equation
- `EPOCHS_END = 15000`
- loss weights in the `weights` dictionary
- residual calibration scales `ion_cont_scale`, `energy_scale`, `ion_mom_scale`

`VETHETA_MAX = 2000` in `model.py` sets the amplitude of the azimuthal velocity envelope.

## Outputs

Written under `output/`:

- `Big_Plots/` — the full field set on one page
- `Individual_Plots/` — one map per field
- `Loss_Plots/` — total and per-term loss against epoch
- `p5_reference_checkpoint.pt` — model weights
- `p5_reference_metrics.json` — converged diagnostics
- `VALIDATION_p5_reference.md` — run report

## Reference values

A converged run at weight seed 7 reaches:

| Quantity | Value |
|---|---|
| `n_i` relative L2 | 0.2048 |
| `T_e` relative L2 | 0.0730 |
| `V_p` relative L2 | 0.0173 |
| Discharge current | 5.394 A |
| `n_i` peak | 0.0780 at x = 0.513 |
| Mean Hall parameter | 119 |
| max abs(`V_etheta`) | 689 |
| Final weighted loss | 7.484 |

The weighted loss is a sum of terms carrying weights that span several orders of magnitude. It
conditions the optimization and is not an accuracy metric; the per-equation unweighted residuals
in the metrics file are.

## Citation

J. Adjedj, *A Self-Consistent Two-Dimensional Physics-Informed Neural Network Model of a Hall
Effect Thruster Plasma Using a Charge-Separation Form of Poisson's Equation*, M.S. thesis,
APRG,
University of Florida, 2026.
