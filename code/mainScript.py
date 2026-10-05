"""P5 reference driver."""

import json
import math
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache")

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np
import torch

from model import PINN
from utils import (
    compute_all_losses,
    compute_collision_frequencies,
    compute_data_loss,
    compute_Si,
    compute_validation_residuals,
    create_output_folders,
    data_loss_l2_relative,
    evaluate_on_grid,
    load_haas_gallimore_data,
    sample_boundary_points,
    sample_domain_points,
    save_combined_plot,
    save_global_metrics,
    save_individual_plots,
    save_loss_plot,
)

torch.set_default_dtype(torch.float64)

LR = 1.0e-4
N_DOMAIN = 10000
N_INLET = 1000
N_WALL = 1000
SEED = 2601
ALPHA_B = 1.0 / 200.0
V24_FINAL_LOSS_REF = 1.2258
LOSS_EXPLOSION_LIMIT = 10.0 * V24_FINAL_LOSS_REF

def build_domain_physics():
    e_phys = 1.602e-19
    eps0 = 8.854e-12
    mi = 2.180e-25
    me = 9.109e-31
    sigma0 = 3.6e-20
    Ei_eV = 12.1
    phi_ref = Ei_eV
    V_ref = math.sqrt(Ei_eV * e_phys / mi)
    Gamma_ref = 1.0e23
    n_ref = Gamma_ref / V_ref
    sigma_ref = sigma0 * math.sqrt(mi / me)
    nu_ref = sigma_ref * Gamma_ref
    L_canal = 0.038
    r_in_phys, r_out_phys = 0.060, 0.085
    S_ref = n_ref * V_ref / L_canal
    k_ref = V_ref / (L_canal * n_ref)
    alpha_p = (e_phys * n_ref * L_canal**2) / (eps0 * phi_ref)
    A_channel = math.pi * (r_out_phys**2 - r_in_phys**2)
    JT_norm = (5.4 / A_channel) / (e_phys * n_ref * V_ref)

    domain = {
        "x_min": 0.0, "x_max": 1.0,
        "r_min": r_in_phys / L_canal,
        "r_max": r_out_phys / L_canal,
        "phi_left": 300.0 / phi_ref,
        "phi_right": 116.2 / phi_ref,
    }
    physics = {
        "e_phys": e_phys,
        "me": me,
        "mi": mi,
        "mn": mi,
        "alpha": alpha_p,
        "n_ref": n_ref,
        "V_ref": V_ref,
        "phi_ref": phi_ref,
        "nu_ref": nu_ref,
        "tau_ref": L_canal / V_ref,
        "S_ref": S_ref,
        "k_ref": k_ref,
        "L_canal": L_canal,
        "eps_i": Ei_eV / phi_ref,
        "Vn": 300.0 / V_ref,
        "delta": 0.1,
        "Vix_inlet": 0.50,
        "nn_exit_target": 0.05,
        "ni_inlet": 0.003,
        "nn_inlet": 1.0,
        "Vetheta_inlet": 0.0,
        "JT_norm": JT_norm,
        "e_over_me": e_phys / me,
        "coulomb_log": 10.0,
        "sigma_en": 2.5e-19,
        "alpha_B": ALPHA_B,
        "Tn_eV": 0.025,
        "Ti_eV": 0.025,
        "energy_alpha": 1.0,
        "energy_scale": 1.0,
        "Te_base": 3.0 / phi_ref,
        "Te_peak": 27.0 / phi_ref,
        "x0_Te": 0.55,
        "sigma_x_Te": 0.15,
        "r0_Te": 0.5 * (r_in_phys + r_out_phys) / L_canal,
        "sigma_r_Te": 0.11842105263157895,
        "B_max": 0.020,
        "x_B": 0.75,
        "sigma_B": 0.30,
        "Di": 0.0,
        "ki_floor_eV": 3.5,
        "ki_softplus_beta": 8.0,
        "ki_scale": 1.0,
    }
    return domain, physics

def build_model(domain, physics, device):
    return PINN(
        input_dim=2,
        output_dim=10,
        hidden_dim=128,
        num_hidden_layers=4,
        phi_left=domain["phi_left"],
        phi_right=domain["phi_right"],
        x_min=domain["x_min"], x_max=domain["x_max"],
        r_min=domain["r_min"], r_max=domain["r_max"],
        delta=physics["delta"],
        Vix_inlet=physics["Vix_inlet"],
        ni_inlet=physics["ni_inlet"],
        nn_inlet=physics["nn_inlet"],
        JT_init=physics["JT_norm"],
        JT_flex=0.3,
        Te_base=physics["Te_base"],
        Te_peak=physics["Te_peak"],
        x0_Te=physics["x0_Te"],
        sigma_x_Te=physics["sigma_x_Te"],
        r0_Te=physics["r0_Te"],
        sigma_r_Te=physics["sigma_r_Te"],
        eta_max=1.0e-4,
        vex_epsilon=0.0,
    ).to(device)

def init_history():
    keys = [
        "total", "poisson", "ion_cont", "neut_cont", "elec_cont", "ion_mom_x",
        "current", "ni_inlet", "nn_inlet", "Vetheta_inlet", "Vir_wall",
        "Ver_wall", "JT_target", "JT_train", "Si_mean", "nn_exit",
        "smooth_ni", "smooth_nn", "eta_abs_mean", "eta_abs_max",
        "ni_mean", "ni_std",
    ]
    return {k: [] for k in keys}

def deterministic_summary(model, domain, physics, history, stop_reason, final_epoch):
    device = next(model.parameters()).device
    residuals = compute_validation_residuals(model, domain, physics, nx=200, nr=80, device=device)

    XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er, eta = evaluate_on_grid(
        model, domain, physics, nx=200, nr=80, device=device
    )
    r_mid = ni.shape[1] // 2
    x = XX[:, r_mid]
    r = RR[0, :]
    ring = 2.0 * math.pi * r
    Lc = physics["L_canal"]
    S_ref = physics["S_ref"]
    n_ref = physics["n_ref"]
    V_ref = physics["V_ref"]
    mi = 2.180e-25

    ni_line = ni[:, r_mid]
    Ex_line = Ex[:, r_mid]
    Te_line = Te[:, r_mid]
    nn_line = nn_[:, r_mid]
    ni_i = int(np.argmax(ni_line))
    ex_i = int(np.argmax(np.abs(Ex_line)))
    te_i = int(np.argmax(Te_line))
    x055_i = int(np.argmin(np.abs(x - 0.55)))

    P_ion = S_ref * np.trapz(np.trapz(Si * ring[None, :], r, axis=1), x) * Lc**3
    C_neut = S_ref * np.trapz(np.trapz((ki * ni * nn_) * ring[None, :], r, axis=1), x) * Lc**3
    F_ion_out = n_ref * V_ref * np.trapz(ni[-1, :] * Vix[-1, :] * ring, r) * Lc**2
    F_n_in = n_ref * V_ref * np.trapz(nn_[0, :] * physics["Vn"] * ring, r) * Lc**2
    F_n_out = n_ref * V_ref * np.trapz(nn_[-1, :] * physics["Vn"] * ring, r) * Lc**2

    JT = model.current_density().detach().cpu().item()
    current_res = ni * Vix - ne * Vex - JT
    eta_max = float(model.eta_max)
    eta_stats = {
        "min": float(np.min(eta)),
        "max": float(np.max(eta)),
        "mean": float(np.mean(eta)),
        "frac_gt_0.5_eta_max": float(np.mean(np.abs(eta) > 0.5 * eta_max)),
        "frac_gt_0.95_eta_max": float(np.mean(np.abs(eta) > 0.95 * eta_max)),
    }
    return {
        "run_name": RUN_NAME,
        "stop_reason": stop_reason,
        "epochs_completed": final_epoch,
        "final_loss": history["total"][-1] if history["total"] else float("nan"),
        "ni_peak_value": float(ni_line[ni_i]),
        "ni_peak_x": float(x[ni_i]),
        "Ex_peak_x": float(x[ex_i]),
        "nn_x055_centerline": float(nn_line[x055_i]),
        "Te_peak_value_norm": float(Te_line[te_i]),
        "Te_peak_value_eV": float(Te_line[te_i] * physics["phi_ref"]),
        "Te_peak_x": float(x[te_i]),
        "IonCont_RMS": residuals["IonCont_RMS"],
        "IonCont_scaled_RMS": residuals["IonCont_scaled_RMS"],
        "NeutCont_RMS": residuals["NeutCont_RMS"],
        "Poisson_RMS": residuals["Poisson_RMS"],
        "ElecCont_RMS": residuals["ElecCont_RMS"],
        "eta_stats": eta_stats,
        "P_ion_particles_s": float(P_ion),
        "C_neut_particles_s": float(C_neut),
        "P_ion_over_C_neut": float(P_ion / C_neut),
        "F_ion_out_particles_s": float(F_ion_out),
        "F_n_in_particles_s": float(F_n_in),
        "F_n_out_particles_s": float(F_n_out),
        "F_ion_out_over_F_n_in": float(F_ion_out / F_n_in),
        "P_ion_kg_s": float(P_ion * mi),
        "C_neut_kg_s": float(C_neut * mi),
        "F_ion_out_kg_s": float(F_ion_out * mi),
        "F_n_in_kg_s": float(F_n_in * mi),
        "JT_residual_RMS": float(np.sqrt(np.mean(current_res**2))),
    }

RUN_NAME            = "p5_reference"
OUTPUT_DIR          = os.path.join(_parent, "output")
CKPT_IN             = ""
CKPT_OUT            = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_checkpoint.pt")
SUMMARY_OUT         = os.path.join(OUTPUT_DIR, f"VALIDATION_{RUN_NAME}.md")

EPOCHS_START        = 0
EPOCHS_END          = 15000
EPOCHS_TO_RUN       = EPOCHS_END - EPOCHS_START
LBFGS_STEPS         = 0

_epochs_override = os.environ.get("EPOCHS_OVERRIDE")
if _epochs_override is not None:
    EPOCHS_TO_RUN = int(_epochs_override)
    print(f"EPOCHS_TO_RUN overridden via env: {EPOCHS_TO_RUN}", flush=True)
_lbfgs_override = os.environ.get("LBFGS_STEPS_OVERRIDE")
if _lbfgs_override is not None:
    LBFGS_STEPS = int(_lbfgs_override)
    print(f"LBFGS_STEPS overridden via env: {LBFGS_STEPS}", flush=True)

CKPT_IN     = os.environ.get("CKPT_IN_OVERRIDE", CKPT_IN)
OUTPUT_DIR  = os.environ.get("OUTPUT_OVERRIDE", OUTPUT_DIR)
CKPT_OUT    = os.environ.get("CKPT_OUT_OVERRIDE", CKPT_OUT)
SUMMARY_OUT = os.environ.get("SUMMARY_OVERRIDE", SUMMARY_OUT)
INIT_SEED   = int(os.environ.get("SEED_OVERRIDE", SEED))
OPTIMIZER_OVERRIDE = os.environ.get("OPTIMIZER_OVERRIDE", "ADAM").strip().upper()
SMOOTH_NI_OVERRIDE = os.environ.get("SMOOTH_NI_OVERRIDE")
_centerline_data_env = os.environ.get("CENTERLINE_DATA_R_MM")
CENTERLINE_DATA_R_MM = (
    float(_centerline_data_env) if _centerline_data_env not in (None, "") else None
)
MONITOR_EVERY       = 500
PRINT_EVERY         = 250

STOP_ETA_SAT_95     = 0.85

STOP_NI_PEAK_MIN    = 0.005

MON_NX = 200
MON_NR = 80

def _grad(u, var):
    return torch.autograd.grad(u.sum(), var, create_graph=True, retain_graph=True)[0]

def report_ni_centerline_init(model, domain, device):
    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_probe = torch.linspace(
        domain["x_min"], domain["x_max"], 11,
        device=device, dtype=torch.float64,
    )
    r_probe = torch.full_like(x_probe, r_center)
    with torch.no_grad():
        _, ni, _, _, _, _, _, _, _, _, _ = model(x_probe, r_probe)
    pairs = [
        f"x={float(x.item()):.2f}:ni={float(n.item()):.5f}"
        for x, n in zip(x_probe, ni)
    ]
    print("Init ni centerline shape (before training/checkpoint load):", flush=True)
    print("  " + " | ".join(pairs), flush=True)

def report_ion_friction_init(model, domain, physics, device):
    x0 = torch.tensor([0.55], device=device, dtype=torch.float64, requires_grad=True)
    r0 = torch.tensor(
        [0.5 * (domain["r_min"] + domain["r_max"])],
        device=device, dtype=torch.float64, requires_grad=True,
    )
    phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, *_ = model(x0, r0)
    phi_x = _grad(phi, x0)
    Si = compute_Si(nn_, ne, Te, physics)
    coll = compute_collision_frequencies(ne, nn_, Te, physics)
    tau_ref = physics.get("tau_ref", physics["L_canal"] / physics["V_ref"])
    nu_in_hat = physics.get("nu_in", 1.8e6) * tau_ref
    ion_factor = (physics["mn"] / physics["mi"]) * nu_in_hat
    electric_x = ni * phi_x
    electron_drag_force_x = ni * (physics["me"] / physics["mi"]) * coll["nu_total"] * tau_ref * (Vex - Vix)
    electron_drag_residual_x = -electron_drag_force_x
    neutral_friction_x = ni * ion_factor * (Vix - physics["Vn"])
    birth_momentum_x = -Si * physics["Vn"]
    total_friction_x = electron_drag_residual_x + neutral_friction_x + birth_momentum_x
    ratio = total_friction_x.detach().abs() / torch.clamp(electric_x.detach().abs(), min=1e-30)
    print(
        "Init conservative ion-friction sign check at x=0.55 centerline: "
        f"Vix={float(Vix.item()):.6g}, Vn={physics['Vn']:.6g}, "
        f"nu_in_hat={nu_in_hat:.6g}, "
        f"neutral_friction_residual_x={float(neutral_friction_x.item()):.6e}, "
        f"birth_momentum_residual_x={float(birth_momentum_x.item()):.6e}, "
        f"electron_drag_residual_x={float(electron_drag_residual_x.item()):.6e}, "
        f"total_drag_residual_x={float(total_friction_x.item()):.6e}, "
        f"ni*dphi/dx={float(electric_x.item()):.6e}, "
        f"|friction/electric|={float(ratio.item()):.6g}. "
        "Positive neutral-friction residual at Vix>Vn corresponds to a negative physical drag force.",
        flush=True,
    )

def report_phi_bc_init(model, domain, device):
    r_probe = torch.linspace(
        domain["r_min"], domain["r_max"], 9,
        device=device, dtype=torch.float64,
    )
    x_left = torch.full_like(r_probe, domain["x_min"])
    x_right = torch.full_like(r_probe, domain["x_max"])
    with torch.no_grad():
        phi_left, *_ = model(x_left, r_probe)
        phi_right, *_ = model(x_right, r_probe)
    left_err = torch.max(torch.abs(phi_left - domain["phi_left"])).item()
    right_err = torch.max(torch.abs(phi_right - domain["phi_right"])).item()
    print(
        "Init phi hard-BC check: "
        f"left={float(phi_left[0]):.12g} target={domain['phi_left']:.12g} "
        f"max_err={left_err:.3e}; "
        f"right={float(phi_right[0]):.12g} target={domain['phi_right']:.12g} "
        f"max_err={right_err:.3e}",
        flush=True,
    )

def report_phi_shape_init(model, domain, physics, device):
    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_probe = torch.linspace(
        domain["x_min"], domain["x_max"], 11,
        device=device, dtype=torch.float64,
    )
    r_probe = torch.full_like(x_probe, r_center)
    xi = (x_probe - domain["x_min"]) / (domain["x_max"] - domain["x_min"])
    phi_linear = domain["phi_left"] + (domain["phi_right"] - domain["phi_left"]) * xi
    cap = xi * (1.0 - xi) * float(model.phi_shape_amp)
    with torch.no_grad():
        phi, _, Vix, _, _, _, _, _, _, _, _ = model(x_probe, r_probe)
    pairs = [
        f"x={float(x.item()):.2f}:phi={float(p.item() * physics['phi_ref']):.1f}V"
        for x, p in zip(x_probe, phi)
    ]
    print("Init phi centerline (A=25 correction active, raw0 at init):", flush=True)
    print("  " + " | ".join(pairs), flush=True)
    idx_055 = int(torch.argmin(torch.abs(x_probe - 0.55)).item())
    print(
        "Phi shape capacity check: "
        f"at x~0.55 linear={float(phi_linear[idx_055] * physics['phi_ref']):.2f}V, "
        f"max correction={float(cap[idx_055] * physics['phi_ref']):.2f}V "
        "(HG needs about +65.6V at x=0.55).",
        flush=True,
    )
    print(
        "Init Vix Bohm-referenced check: "
        f"Vix(x=0)={float(Vix[0]):.6g}, Vix(x~0.55)={float(Vix[idx_055]):.6g}, "
        f"Vix(exit)={float(Vix[-1]):.6g}, "
        f"anode target={getattr(model, 'Vix_anode_min', physics['Vix_inlet']):.6g}, "
        f"Bohm-criterion speed={physics['Vix_inlet']:.6g}.",
        flush=True,
    )

def _evaluate_monitor(model, domain, physics, device):
    """Run the 4 monitored quantities on a deterministic 200x80 grid."""

    r_center = 0.5 * (domain["r_min"] + domain["r_max"])
    x_cl = torch.linspace(
        domain["x_min"], domain["x_max"], MON_NX,
        device=device, dtype=torch.float64,
    )
    r_cl = torch.full_like(x_cl, r_center)
    with torch.no_grad():
        _, ni_cl, _, _, _, _, _, _, _, _, _ = model(x_cl, r_cl)
    ni_cl_np = ni_cl.cpu().numpy()
    ni_peak_idx = int(np.argmax(ni_cl_np))
    ni_peak_value = float(ni_cl_np[ni_peak_idx])
    ni_peak_x = float(x_cl[ni_peak_idx].item())

    x_lin = torch.linspace(
        domain["x_min"], domain["x_max"], MON_NX,
        device=device, dtype=torch.float64,
    )
    r_lin = torch.linspace(
        domain["r_min"], domain["r_max"], MON_NR,
        device=device, dtype=torch.float64,
    )
    XX, RR = torch.meshgrid(x_lin, r_lin, indexing="ij")
    with torch.no_grad():
        _, _, _, _, _, _, _, _, _, _, eta = model(XX.reshape(-1), RR.reshape(-1))
    eta_np = eta.cpu().numpy().reshape(MON_NX, MON_NR)
    eta_max_v = float(model.eta_max)
    eta_abs = np.abs(eta_np)
    sat_frac_95 = float(np.mean(eta_abs > 0.95 * eta_max_v))
    sat_frac_50 = float(np.mean(eta_abs > 0.5  * eta_max_v))

    residuals = compute_validation_residuals(
        model, domain, physics, nx=MON_NX, nr=MON_NR, device=device
    )

    return dict(
        ni_peak_value=ni_peak_value,
        ni_peak_x=ni_peak_x,
        sat_frac_95=sat_frac_95,
        sat_frac_50=sat_frac_50,
        eta_min=float(eta_np.min()),
        eta_max_val=float(eta_np.max()),
        eta_mean=float(eta_np.mean()),
        IonCont_RMS=residuals["IonCont_RMS"],
        NeutCont_RMS=residuals["NeutCont_RMS"],
        Poisson_RMS=residuals["Poisson_RMS"],
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INIT_SEED)
    print(f"Using device: {device}", flush=True)
    print(
        f"Init seed before build_model = {INIT_SEED}; "
        f"collocation/boundary seed = {SEED}.",
        flush=True,
    )
    print(
        f"Training {RUN_NAME}: from_scratch={not bool(CKPT_IN)}, checkpoint={CKPT_IN or '<none>'}; "
        f"epoch {EPOCHS_START} -> {EPOCHS_START + EPOCHS_TO_RUN}. "
        f"{OPTIMIZER_OVERRIDE} epochs followed by {LBFGS_STEPS} L-BFGS steps.",
        flush=True,
    )

    domain, physics = build_domain_physics()
    physics["ni_inlet"] = 4.64e17 / physics["n_ref"]
    physics["Vn"] = 450.0 / physics["V_ref"]
    physics["nn_exit_target"] = -1.0
    physics["ki_taper_center_eV"] = 5.0
    physics["ki_taper_width_eV"] = 0.35
    physics["nu_in"] = 1.80e6
    physics["energy_alpha"] = 2.5
    e_phys = physics["e_phys"]
    L_canal = physics["L_canal"]
    r_in_phys = domain["r_min"] * L_canal
    r_out_phys = domain["r_max"] * L_canal
    A_channel = math.pi * (r_out_phys**2 - r_in_phys**2)
    physics["A_channel"] = A_channel
    physics["JT_target_A"] = 5.4
    physics["JT_A_scale"] = e_phys * physics["n_ref"] * physics["V_ref"] * A_channel
    physics["JT_norm"] = physics["JT_target_A"] / physics["JT_A_scale"]
    print(
        f"Closures: ni_inlet={physics['ni_inlet']:.6g} "
        f"({physics['ni_inlet'] * physics['n_ref']:.3e} m^-3), "
        f"Vn={physics['Vn'] * physics['V_ref']:.1f} m/s, "
        f"nu_in={physics['nu_in']:.3e} s^-1 "
        f"(nu_in_hat={physics['nu_in'] * physics['tau_ref']:.6g}), "
        f"ki taper center={physics['ki_taper_center_eV']:.2f} eV, "
        f"width={physics['ki_taper_width_eV']:.2f} eV, "
        f"alpha_B={ALPHA_B:.6g}, energy_alpha={physics['energy_alpha']:.2f}.",
        flush=True,
    )
    print(
        f"Discharge-current target: Id={physics['JT_target_A']:.3f} A, "
        f"A_channel={A_channel:.6e} m^2, JT_norm={physics['JT_norm']:.8g}, "
        f"JT_A_scale={physics['JT_A_scale']:.6g} A per normalized JT.",
        flush=True,
    )
    create_output_folders(OUTPUT_DIR)

    data_cfg = {"enabled": True, "weight_ni": 0.0, "weight_Te": 0.0, "weight_Vp": 0.0}
    cfg_path = os.path.join(_here, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                data_cfg = json.load(f).get("data_loss", {}) or {}
        except Exception as e:
            print(f"WARN: failed to parse config.json data_loss section: {e}",
                  flush=True)
            data_cfg = {}
    data_enabled = bool(data_cfg.get("enabled", True))
    exp_data = None
    exp_data_all = None
    w_data = {"ni": 1.0, "Te": 1.0, "Vp": 1.0}
    if data_enabled:
        w_data = {
            "ni": float(data_cfg.get("weight_ni", 1.0)),
            "Te": float(data_cfg.get("weight_Te", 1.0)),
            "Vp": float(data_cfg.get("weight_Vp", 1.0)),
        }
        csv_path = data_cfg.get("csv_path", "haas_gallimore_p5_5p4A.csv")
        if not os.path.isabs(csv_path):
            for base in (_here, _parent, os.getcwd()):
                cand = os.path.normpath(os.path.join(base, csv_path))
                if os.path.exists(cand):
                    csv_path = cand
                    break
        if not os.path.exists(csv_path):
            print(f"WARN: data_loss.csv_path not found ({csv_path}); "
                  "disabling data loss.", flush=True)
            data_enabled = False
        else:
            print(f"Hybrid data loss ENABLED (weights ni={w_data['ni']}, "
                  f"Te={w_data['Te']}, Vp={w_data['Vp']}); csv={csv_path}",
                  flush=True)
            exp_data_all = load_haas_gallimore_data(
                csv_path, domain, physics, device, dtype=torch.float64,
            )
            if CENTERLINE_DATA_R_MM is None:
                exp_data = exp_data_all
            else:
                print(
                    f"Training data loss restricted to r_mm={CENTERLINE_DATA_R_MM:g}; "
                    "other Haas/Gallimore radii are held out for evaluation only.",
                    flush=True,
                )
                exp_data = load_haas_gallimore_data(
                    csv_path, domain, physics, device, dtype=torch.float64,
                    include_r_mm=CENTERLINE_DATA_R_MM,
                )
    else:
        print("Hybrid data loss DISABLED (config.data_loss.enabled=false "
              "or section missing).", flush=True)
    base_w_data = dict(w_data)
    data_weight_scale = 1.0
    data_anneal_schedule = [
        {"start_epoch": 0, "end_epoch": EPOCHS_TO_RUN, "scale": 1.0},
    ]
    data_anneal_events = []

    def set_data_weight_scale(scale, epoch, reason):
        nonlocal data_weight_scale, w_data
        data_weight_scale = scale
        w_data = {k: base_w_data[k] * data_weight_scale for k in base_w_data}
        event = {
            "epoch": int(epoch),
            "scale": float(data_weight_scale),
            "weights": {k: float(v) for k, v in w_data.items()},
            "reason": reason,
        }
        data_anneal_events.append(event)
        print(
            f"Data-weight anneal @ epoch {epoch}: scale={data_weight_scale:.3g}, "
            f"weights ni/Te/Vp={w_data['ni']:.6g}/{w_data['Te']:.6g}/{w_data['Vp']:.6g} "
            f"({reason})",
            flush=True,
        )

    def annotate_data_weights(snap):
        snap["data_weight_scale"] = float(data_weight_scale)
        snap["data_weight_ni"] = float(w_data.get("ni", 0.0))
        snap["data_weight_Te"] = float(w_data.get("Te", 0.0))
        snap["data_weight_Vp"] = float(w_data.get("Vp", 0.0))

    if data_enabled and exp_data is not None:
        print(
            f"Data weights fixed for this run: ni/Te/Vp="
            f"{w_data['ni']:.6g}/{w_data['Te']:.6g}/{w_data['Vp']:.6g}; no annealing.",
            flush=True,
        )


    model = build_model(domain, physics, device)
    ckpt_in_data = None
    if CKPT_IN:
        ckpt_in_data = torch.load(CKPT_IN, map_location=device)
        model.load_state_dict(ckpt_in_data["model_state_dict"], strict=True)
        print(
            f"Loaded checkpoint weights from {CKPT_IN}; "
            f"checkpoint epochs_completed={ckpt_in_data.get('epochs_completed', 'unknown')}.",
            flush=True,
        )
    physics["Vix_anode_min"] = float(getattr(model, "Vix_anode_min", physics["Vix_inlet"]))
    physics["ni_seed_peak"] = float(getattr(model, "ni_seed_peak", float("nan")))
    physics["ni_seed_amp"] = float(getattr(model, "ni_seed_amp", float("nan")))
    print("Output index mapping: 0=phi, 1=ni, 2=Vix, 3=Vir, 4=nn, "
          "5=ne, 6=Vex, 7=Ver, 8=Te, 9=Vetheta, 10=eta", flush=True)
    print(
        "Network raw outputs: 0=phi, 1=direct ni log-amplitude, 2=Vex correction, "
        "3=Vir, 4=nn depletion, 5=Ver, 6=Vetheta, 7=eta, 8=Te, 9=Vix.",
        flush=True,
    )
    print(
        "Phi form: phi = phi_L + (phi_R - phi_L)*xi "
        "+ xi*(1-xi)*25*tanh(raw0); end Dirichlet BCs are hard.",
        flush=True,
    )
    report_phi_bc_init(model, domain, device)
    report_phi_shape_init(model, domain, physics, device)
    report_ni_centerline_init(model, domain, device)
    report_ion_friction_init(model, domain, physics, device)
    print(f"Model initialized. eta_max={float(model.eta_max):.2e}, "
          f"ki_scale={physics['ki_scale']}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_f, r_f = sample_domain_points(N_DOMAIN, domain, device)
    boundary_data = sample_boundary_points(N_INLET, N_WALL, domain, device)

    weights = {
        "poisson": 100.0,
        "ion_cont": 200.0,
        "neut_cont": 100.0,
        "elec_cont": 10.0,
        "energy": 1.0,
        "ion_mom_x": 20.0,
        "ion_mom_r": 20.0,
        "ni_inlet": 0.0,
        "nn_inlet": 200.0,
        "Vetheta_inlet": 20.0,
        "Vir_wall": 20.0,
        "Ver_wall": 10.0,
        "JT_target": 100.0,
        "JT_current_A": 5.0,
        "nn_exit": 100.0,
        "smooth_ni": 1.0e-2,
        "smooth_nn": 1.0e-2,
        "smooth_phi": 1.0e-1,
        "smooth_Er": 0.0,
        "smooth_Te": 1.0e-3,
        "elec_mom_x": 1.0e-2,
        "elec_mom_r": 1.0e-2,
        "elec_mom_theta": 1.0e-2,
    }
    if SMOOTH_NI_OVERRIDE not in (None, ""):
        weights["smooth_ni"] = float(SMOOTH_NI_OVERRIDE)
        print(f"smooth_ni overridden via env: {weights['smooth_ni']:.6g}", flush=True)



    _, scale_probe = compute_all_losses(model, x_f, r_f, boundary_data, domain, physics, weights)
    restored_scale_keys = []
    if CKPT_IN and ckpt_in_data is not None:
        ckpt_physics = ckpt_in_data.get("physics", {}) or {}
        for scale_key in ("ion_cont_scale", "energy_scale", "ion_mom_scale"):
            if scale_key in ckpt_physics:
                physics[scale_key] = float(ckpt_physics[scale_key])
                restored_scale_keys.append(scale_key)
    if len(restored_scale_keys) == 3:
        print(
            "Restored residual scale factors from checkpoint: "
            f"ion_cont={physics['ion_cont_scale']:.6e}, "
            f"energy={physics['energy_scale']:.6e}, "
            f"ion_mom={physics['ion_mom_scale']:.6e}.",
            flush=True,
        )
    else:
        ion_cont_rms = math.sqrt(max(scale_probe.get("ion_cont_raw", scale_probe.get("ion_cont", 1.0)), 1e-12))
        energy_rms = math.sqrt(max(scale_probe.get("energy", 1.0), 1e-12))
        ion_mom_rms = math.sqrt(max(
            scale_probe.get("ion_mom_x", 1.0) + scale_probe.get("ion_mom_r", 0.0),
            1e-12,
        ))
        physics["ion_cont_scale"] = ion_cont_rms
        physics["energy_scale"] = energy_rms
        physics["ion_mom_scale"] = ion_mom_rms
        print(f"Ion continuity residual scale factor: {ion_cont_rms:.6e}", flush=True)
        print(f"Electron energy residual scale factor: {energy_rms:.6e}", flush=True)
        print(f"Ion momentum residual scale factor: {ion_mom_rms:.6e}", flush=True)
    physics["ion_cont_scale"] = 0.6100892858913114
    physics["energy_scale"] = 0.965239365788521
    physics["ion_mom_scale"] = 0.7936910647527818
    print(
        "Fixed reference residual scales: "
        f"ion_cont={physics['ion_cont_scale']:.16g}, "
        f"energy={physics['energy_scale']:.16g}, "
        f"ion_mom={physics['ion_mom_scale']:.16g}.",
        flush=True,
    )
    print(
        "Ion force sign check: the one-dimensional closure gives Vix*dVix/dx + dphi/dx = 0; "
        "therefore conservative residual uses +ni*dphi/dx.",
        flush=True,
    )
    print(f"Typical collision magnitudes at init: nu_ei={scale_probe.get('nu_ei_mean_Hz', float('nan')):.3e} Hz, "
          f"nu_en={scale_probe.get('nu_en_mean_Hz', float('nan')):.3e} Hz", flush=True)


    optimizer_hyperparams = {}
    if OPTIMIZER_OVERRIDE == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer_hyperparams = {"lr": float(LR)}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=300, min_lr=1e-6
        )
    elif OPTIMIZER_OVERRIDE == "SOAP":
        from soap import SOAP
        soap_kwargs = {
            "lr": 3.0e-3,
            "betas": (0.95, 0.95),
            "weight_decay": 0.01,
            "precondition_frequency": 10,
        }
        optimizer = SOAP(model.parameters(), **soap_kwargs)
        optimizer_hyperparams = {
            "lr": soap_kwargs["lr"],
            "betas": list(soap_kwargs["betas"]),
            "weight_decay": soap_kwargs["weight_decay"],
            "precondition_frequency": soap_kwargs["precondition_frequency"],
            "source": "nikhilvyas/SOAP soap.py defaults/recommended README example",
        }
        scheduler = None
        print(f"SOAP optimizer enabled with hyperparameters: {optimizer_hyperparams}", flush=True)
    else:
        raise ValueError(f"Unsupported OPTIMIZER_OVERRIDE={OPTIMIZER_OVERRIDE!r}")

    history = init_history()

    for _k in (
        "L_data_ni", "L_data_Te", "L_data_Vp", "L_data_total",
        "energy", "ion_mom_r", "elec_mom_x", "elec_mom_r", "elec_mom_theta",
        "JT_current_A", "JT_train_A", "smooth_phi", "smooth_Er", "smooth_Te",
    ):
        history[_k] = []
    stop_reason = f"completed {EPOCHS_START + EPOCHS_TO_RUN} {OPTIMIZER_OVERRIDE} epochs + {LBFGS_STEPS} L-BFGS steps"
    final_epoch = EPOCHS_START

    def add_hybrid_data_loss(total, loss_dict):
        if data_enabled and exp_data is not None:
            data_losses = compute_data_loss(model, exp_data)
            weighted_data = (
                w_data["ni"] * data_losses["ni"]
                + w_data["Te"] * data_losses["Te"]
                + w_data["Vp"] * data_losses["Vp"]
            )
            total = total + weighted_data
            loss_dict["L_data_ni"] = float(data_losses["ni"].detach().item())
            loss_dict["L_data_Te"] = float(data_losses["Te"].detach().item())
            loss_dict["L_data_Vp"] = float(data_losses["Vp"].detach().item())
            loss_dict["L_data_total"] = float(weighted_data.detach().item())
            loss_dict["total"] = float(total.detach().item())
        else:
            loss_dict["L_data_ni"] = 0.0
            loss_dict["L_data_Te"] = 0.0
            loss_dict["L_data_Vp"] = 0.0
            loss_dict["L_data_total"] = 0.0
        return total, loss_dict

    trajectory = []
    start_snap = _evaluate_monitor(model, domain, physics, device)
    start_snap["epoch"] = EPOCHS_START
    init_total, init_loss_dict = compute_all_losses(model, x_f, r_f, boundary_data, domain, physics, weights)
    init_total, init_loss_dict = add_hybrid_data_loss(init_total, init_loss_dict)
    start_snap["total_loss"] = float(init_loss_dict["total"])
    annotate_data_weights(start_snap)
    loss_explosion_limit = max(LOSS_EXPLOSION_LIMIT, 10.0 * float(init_loss_dict["total"]))
    data_initial = None
    if data_enabled and exp_data is not None:
        init_data_losses = compute_data_loss(model, exp_data)
        data_initial = {
            "losses": {k: float(v.detach().item()) for k, v in init_data_losses.items()},
            "l2": data_loss_l2_relative(model, exp_data),
        }
    trajectory.append(start_snap)
    print(
        f"[epoch {EPOCHS_START:5d}] total={start_snap['total_loss']:.6f}  "
        f"ni_peak={start_snap['ni_peak_value']:.4f}@x={start_snap['ni_peak_x']:.3f}  "
        f"sat95={start_snap['sat_frac_95']:.3f}  "
        f"IonCont={start_snap['IonCont_RMS']:.4f} "
        f"NeutCont={start_snap['NeutCont_RMS']:.4f} "
        f"Poisson={start_snap['Poisson_RMS']:.4f} "
        f"wdata={data_weight_scale:.3g}",
        flush=True,
    )

    t0 = time.time()
    early_stop = False
    last_loss_dict = dict(init_loss_dict)

    def weighted_loss_breakdown(loss_dict):
        terms = {}
        for key, weight_key in (
            ("poisson", "poisson"),
            ("ion_cont", "ion_cont"),
            ("neut_cont", "neut_cont"),
            ("elec_cont", "elec_cont"),
            ("energy", "energy"),
            ("ion_mom_x", "ion_mom_x"),
            ("ion_mom_r", "ion_mom_r"),
            ("elec_mom_x", "elec_mom_x"),
            ("elec_mom_r", "elec_mom_r"),
            ("elec_mom_theta", "elec_mom_theta"),
            ("ni_inlet", "ni_inlet"),
            ("nn_inlet", "nn_inlet"),
            ("nn_exit", "nn_exit"),
            ("Vetheta_inlet", "Vetheta_inlet"),
            ("Vir_wall", "Vir_wall"),
            ("Ver_wall", "Ver_wall"),
            ("JT_target", "JT_target"),
            ("JT_current_A", "JT_current_A"),
            ("smooth_ni", "smooth_ni"),
            ("smooth_nn", "smooth_nn"),
            ("smooth_phi", "smooth_phi"),
            ("smooth_Er", "smooth_Er"),
            ("smooth_Te", "smooth_Te"),
        ):
            terms[key] = float(weights.get(weight_key, 0.0) * loss_dict.get(key, 0.0))
        terms["data_ni"] = float(w_data.get("ni", 0.0) * loss_dict.get("L_data_ni", 0.0))
        terms["data_Te"] = float(w_data.get("Te", 0.0) * loss_dict.get("L_data_Te", 0.0))
        terms["data_Vp"] = float(w_data.get("Vp", 0.0) * loss_dict.get("L_data_Vp", 0.0))
        terms["data_total"] = float(loss_dict.get("L_data_total", 0.0))
        terms["total_from_terms"] = float(
            sum(v for k, v in terms.items() if k not in ("data_total", "total_from_terms"))
        )
        terms["reported_total"] = float(loss_dict.get("total", float("nan")))
        return terms

    def annotate_weighted_losses(snap, loss_dict):
        terms = weighted_loss_breakdown(loss_dict)
        snap["weighted_loss_terms"] = terms
        snap["weighted_loss_terms_sorted"] = [
            {"term": k, "weighted_value": v}
            for k, v in sorted(
                ((k, v) for k, v in terms.items() if k not in ("total_from_terms", "reported_total")),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        ]

    def append_history(loss_dict):
        for hk in history:
            if hk not in ("ni_mean", "ni_std") and hk in loss_dict:
                history[hk].append(loss_dict[hk])
        history["ni_mean"].append(
            history["ni_mean"][-1] if history["ni_mean"] else 0.0
        )
        history["ni_std"].append(
            history["ni_std"][-1] if history["ni_std"] else 0.0
        )

    for k in range(EPOCHS_TO_RUN):
        epoch = EPOCHS_START + k + 1
        optimizer.zero_grad()
        total, loss_dict = compute_all_losses(
            model, x_f, r_f, boundary_data, domain, physics, weights
        )

        total, loss_dict = add_hybrid_data_loss(total, loss_dict)
        last_loss_dict = dict(loss_dict)
        if not torch.isfinite(total):
            stop_reason = f"NaN/Inf loss at epoch {epoch}"
            early_stop = True
            final_epoch = epoch
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss_dict["total"])

        append_history(loss_dict)

        final_epoch = epoch

        if epoch % PRINT_EVERY == 0 and epoch % MONITOR_EVERY != 0:
            with torch.no_grad():
                _, ni_c, _, _, _, _, _, _, _, _, _ = model(x_f.detach(), r_f.detach())
                xi_check = (x_f.detach() - domain["x_min"]) / (
                    domain["x_max"] - domain["x_min"]
                )
                interior_mask = xi_check > 0.05
                ni_int_max = (
                    ni_c[interior_mask].max().item()
                    if interior_mask.any() else ni_c.mean().item()
                )
            data_str = (
                f" | Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                if data_enabled else ""
            )
            print(
                f"  epoch {epoch:5d} | total={loss_dict['total']:.4f} | "
                f"Poisson={loss_dict['poisson']:.3e} | "
                f"IonCont={loss_dict['ion_cont']:.4f} | "
                f"NeutCont={loss_dict['neut_cont']:.4f} | "
                f"|eta|={loss_dict['eta_abs_mean']:.2e} | "
                f"JT*={loss_dict['JT_train']:.4f} | "
                f"ni int_max={ni_int_max:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"wdata={data_weight_scale:.3g}"
                + data_str,
                flush=True,
            )

        if epoch % MONITOR_EVERY == 0:
            snap = _evaluate_monitor(model, domain, physics, device)
            snap["epoch"] = epoch
            snap["total_loss"] = float(loss_dict["total"])
            annotate_data_weights(snap)
            annotate_weighted_losses(snap, loss_dict)
            trajectory.append(snap)
            data_str = (
                f"  Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                if data_enabled else ""
            )
            print(
                f"[epoch {epoch:5d}] total={snap['total_loss']:.6f}  "
                f"ni_peak={snap['ni_peak_value']:.4f}@x={snap['ni_peak_x']:.3f}  "
                f"sat95={snap['sat_frac_95']:.3f}  "
                f"IonCont={snap['IonCont_RMS']:.4f} "
                f"NeutCont={snap['NeutCont_RMS']:.4f} "
                f"Poisson={snap['Poisson_RMS']:.4f}"
                f"  wdata={data_weight_scale:.3g}"
                + data_str,
                flush=True,
            )

            if snap["sat_frac_95"] > STOP_ETA_SAT_95:
                stop_reason = (
                    f"eta-saturation stop at epoch {epoch}: "
                    f"sat_frac_95={snap['sat_frac_95']:.4f} > {STOP_ETA_SAT_95}"
                )
                early_stop = True
                break
            if snap["ni_peak_value"] < STOP_NI_PEAK_MIN:
                stop_reason = (
                    f"ni-peak collapse at epoch {epoch}: "
                    f"ni_peak={snap['ni_peak_value']:.4f} < {STOP_NI_PEAK_MIN}"
                )
                early_stop = True
                break



        pde_total = loss_dict["total"] - loss_dict.get("L_data_total", 0.0)
        if pde_total > loss_explosion_limit:
            stop_reason = (
                f"loss explosion at epoch {epoch}: "
                f"pde={pde_total:.6g} > {loss_explosion_limit:.6g}"
            )
            early_stop = True
            break

    if (not early_stop) and LBFGS_STEPS > 0:
        print(f"Starting L-BFGS refinement for {LBFGS_STEPS} steps.", flush=True)
        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=1,
            max_eval=5,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        lbfgs_last = {"loss": None, "loss_dict": dict(last_loss_dict)}

        for j in range(LBFGS_STEPS):
            epoch = EPOCHS_START + EPOCHS_TO_RUN + j + 1

            def closure():
                lbfgs_optimizer.zero_grad()
                total_c, loss_dict_c = compute_all_losses(
                    model, x_f, r_f, boundary_data, domain, physics, weights
                )
                total_c, loss_dict_c = add_hybrid_data_loss(total_c, loss_dict_c)
                if torch.isfinite(total_c):
                    total_c.backward()
                lbfgs_last["loss"] = total_c.detach()
                lbfgs_last["loss_dict"] = dict(loss_dict_c)
                return total_c

            lbfgs_optimizer.step(closure)
            loss_dict = dict(lbfgs_last["loss_dict"])
            total_value = float(loss_dict.get("total", float("nan")))
            last_loss_dict = dict(loss_dict)
            append_history(loss_dict)
            final_epoch = epoch

            if (not math.isfinite(total_value)):
                stop_reason = f"NaN/Inf loss during L-BFGS at step {j + 1}"
                early_stop = True
                break

            if epoch % PRINT_EVERY == 0 and epoch % MONITOR_EVERY != 0:
                with torch.no_grad():
                    _, ni_c, _, _, _, _, _, _, _, _, _ = model(x_f.detach(), r_f.detach())
                    xi_check = (x_f.detach() - domain["x_min"]) / (
                        domain["x_max"] - domain["x_min"]
                    )
                    interior_mask = xi_check > 0.05
                    ni_int_max = (
                        ni_c[interior_mask].max().item()
                        if interior_mask.any() else ni_c.mean().item()
                    )
                data_str = (
                    f" | Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                    f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                    if data_enabled else ""
                )
                print(
                    f"  lbfgs {j + 1:5d} (epoch {epoch:5d}) | total={total_value:.4f} | "
                    f"Poisson={loss_dict['poisson']:.3e} | "
                    f"IonCont={loss_dict['ion_cont']:.4f} | "
                    f"NeutCont={loss_dict['neut_cont']:.4f} | "
                    f"|eta|={loss_dict['eta_abs_mean']:.2e} | "
                    f"JT*={loss_dict['JT_train']:.4f} | "
                    f"ni int_max={ni_int_max:.4f}"
                    + data_str,
                    flush=True,
                )

            if epoch % MONITOR_EVERY == 0:
                snap = _evaluate_monitor(model, domain, physics, device)
                snap["epoch"] = epoch
                snap["total_loss"] = total_value
                annotate_data_weights(snap)
                annotate_weighted_losses(snap, loss_dict)
                trajectory.append(snap)
                data_str = (
                    f"  Ldata(ni/Te/Vp)={loss_dict['L_data_ni']:.3e}/"
                    f"{loss_dict['L_data_Te']:.3e}/{loss_dict['L_data_Vp']:.3e}"
                    if data_enabled else ""
                )
                print(
                    f"[lbfgs {j + 1:5d} | epoch {epoch:5d}] total={total_value:.6f}  "
                    f"ni_peak={snap['ni_peak_value']:.4f}@x={snap['ni_peak_x']:.3f}  "
                    f"sat95={snap['sat_frac_95']:.3f}  "
                    f"IonCont={snap['IonCont_RMS']:.4f} "
                    f"NeutCont={snap['NeutCont_RMS']:.4f} "
                    f"Poisson={snap['Poisson_RMS']:.4f}"
                    + data_str,
                    flush=True,
                )
                if snap["sat_frac_95"] > STOP_ETA_SAT_95:
                    stop_reason = (
                        f"eta-saturation stop during L-BFGS step {j + 1}: "
                        f"sat_frac_95={snap['sat_frac_95']:.4f} > {STOP_ETA_SAT_95}"
                    )
                    early_stop = True
                    break
                if snap["ni_peak_value"] < STOP_NI_PEAK_MIN:
                    stop_reason = (
                        f"ni-peak collapse during L-BFGS step {j + 1}: "
                        f"ni_peak={snap['ni_peak_value']:.4f} < {STOP_NI_PEAK_MIN}"
                    )
                    early_stop = True
                    break

            pde_total = loss_dict["total"] - loss_dict.get("L_data_total", 0.0)
            if pde_total > loss_explosion_limit:
                stop_reason = (
                    f"loss explosion during L-BFGS step {j + 1}: "
                    f"pde={pde_total:.6g} > {loss_explosion_limit:.6g}"
                )
                early_stop = True
                break

    elapsed = time.time() - t0
    print(f"Training stopped: {stop_reason}", flush=True)
    print(f"Elapsed: {elapsed/60.0:.2f} min", flush=True)

    model.eval()
    physics_out = dict(physics)
    physics_out["JT_trained"] = model.current_density().detach().cpu().item()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "physics": physics_out,
            "domain": domain,
            "eta_max": float(model.eta_max),
            "run_name": RUN_NAME,
            "epochs_completed": final_epoch,
            "adam_epochs_requested": EPOCHS_TO_RUN if OPTIMIZER_OVERRIDE == "ADAM" else 0,
            "optimizer_name": OPTIMIZER_OVERRIDE,
            "optimizer_steps_requested": EPOCHS_TO_RUN,
            "optimizer_hyperparams": optimizer_hyperparams,
            "lbfgs_steps_requested": LBFGS_STEPS,
            "stop_reason": stop_reason,
            "final_loss": history["total"][-1] if history["total"] else float("nan"),
            "trajectory": trajectory,
            "data_anneal": {
                "scheme": "fixed_full_data_no_annealing",
                "schedule": data_anneal_schedule,
                "base_weights": {k: float(v) for k, v in base_w_data.items()},
                "events": data_anneal_events,
                "final_scale": float(data_weight_scale),
                "final_weights": {k: float(v) for k, v in w_data.items()},
                "centerline_data_r_mm": CENTERLINE_DATA_R_MM,
            },
            "resume_from": CKPT_IN,
        },
        CKPT_OUT,
    )
    print(f"Checkpoint saved: {CKPT_OUT}", flush=True)

    XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er, eta = (
        evaluate_on_grid(model, domain, physics_out, nx=200, nr=200, device=device)
    )
    save_combined_plot(
        XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er,
        eta, OUTPUT_DIR, RUN_NAME, physics_out,
    )
    save_individual_plots(
        XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er,
        eta, OUTPUT_DIR, RUN_NAME,
    )
    save_loss_plot(history, OUTPUT_DIR, RUN_NAME, physics_out)
    save_global_metrics(model, domain, physics_out, OUTPUT_DIR, RUN_NAME,
                        nr=200, device=device)

    data_final = None
    if data_enabled and exp_data is not None:
        l2_rel = data_loss_l2_relative(model, exp_data)
        final_ni = history["L_data_ni"][-1] if history["L_data_ni"] else float("nan")
        final_Te = history["L_data_Te"][-1] if history["L_data_Te"] else float("nan")
        final_Vp = history["L_data_Vp"][-1] if history["L_data_Vp"] else float("nan")
        data_final = {
            "losses": {
                "ni": final_ni,
                "Te": final_Te,
                "Vp": final_Vp,
                "total": history["L_data_total"][-1] if history["L_data_total"] else float("nan"),
            },
            "l2": l2_rel,
        }
        print("\n=== Hybrid data-loss final summary (Haas & Gallimore P5) ===",
              flush=True)
        print(f"  L_data_ni (log-MSE, normalized density) = {final_ni:.6e}", flush=True)
        print(f"  L_data_Te (MSE, normalized) = {final_Te:.6e}", flush=True)
        print(f"  L_data_Vp (MSE, normalized) = {final_Vp:.6e}", flush=True)
        print(f"  L2 relative error  ni = {l2_rel.get('ni', float('nan')):.4f}",
              flush=True)
        print(f"  L2 relative error  Te = {l2_rel.get('Te', float('nan')):.4f}",
              flush=True)
        print(f"  L2 relative error  Vp = {l2_rel.get('Vp', float('nan')):.4f}",
              flush=True)
        print("  (Te is a trainable output in v3; Te data loss back-propagates.)",
              flush=True)

    summary = deterministic_summary(
        model, domain, physics_out, history, stop_reason, final_epoch
    )
    _write_validation_report(
        summary, trajectory, stop_reason, final_epoch, early_stop,
        init_loss_dict, last_loss_dict, physics_out, weights,
        data_initial, data_final, w_data, loss_explosion_limit,
    )

def _write_validation_report(
    summary, trajectory, stop_reason, final_epoch, early_stop,
    init_loss, final_loss, physics, weights, data_initial, data_final,
    data_weights, loss_explosion_limit,
):
    """Write VALIDATION_V3.md for the fully physics-based V3 run."""
    s = summary
    lines = [
        "# V3 validation report",
        "",
        f"Run name: {RUN_NAME}",
        f"Training history: {stop_reason}. Final optimizer epoch/step index = {final_epoch}.",
        f"Requested schedule: Adam epochs = {EPOCHS_TO_RUN}; L-BFGS steps = {LBFGS_STEPS}.",
        f"Final total loss = {s['final_loss']:.6g}. Early stop = {early_stop}.",
        f"Loss explosion guard used = {loss_explosion_limit:.6g}.",
        "",
        "## Output index mapping",
        "",
        "Forward returns: 0=phi, 1=ni, 2=Vix, 3=Vir, 4=nn, 5=ne, "
        "6=Vex, 7=Ver, 8=Te, 9=Vetheta, 10=eta.",
        "Network raw outputs: 0=phi correction, 1=direct ni log-amplitude, 2=Vex correction, "
        "3=Vir, 4=nn depletion, 5=Ver, 6=Vetheta, 7=eta, 8=Te, 9=Vix.",
        "",
        "## Normalization and constants",
        "",
        f"- phi_ref = {physics['phi_ref']:.6g} eV.",
        f"- nu_ref = {physics['nu_ref']:.6e} 1/s.",
        f"- tau_ref = L_canal/V_ref = {physics['tau_ref']:.6e} s.",
        f"- ion continuity residual scale = {physics.get('ion_cont_scale', 1.0):.6e}.",
        f"- electron energy residual scale = {physics['energy_scale']:.6e}.",
        f"- ion momentum residual scale = {physics.get('ion_mom_scale', 1.0):.6e}.",
        f"- ion-neutral collision frequency nu_in = {physics.get('nu_in', 1.8e6):.6e} s^-1; "
        f"nu_in_hat = nu_in*tau_ref = {physics.get('nu_in', 1.8e6) * physics['tau_ref']:.6g}.",
        f"- discharge-current target = {physics.get('JT_target_A', float('nan')):.6g} A; "
        f"A_channel = {physics.get('A_channel', float('nan')):.6e} m^2; "
        f"JT_norm = {physics['JT_norm']:.8g}; "
        f"JT_A_scale = {physics.get('JT_A_scale', float('nan')):.6g} A per normalized JT.",
        f"- explicit current penalty: L_current = "
        f"(JT_hat*JT_A_scale - {physics.get('JT_target_A', float('nan')):.6g} A)^2, "
        f"weight = {weights.get('JT_current_A', 0.0):.6g}.",
        f"- ki low-Te taper = sigmoid((Te - {physics.get('ki_taper_center_eV', 5.0):.3g} eV) / "
        f"{physics.get('ki_taper_width_eV', 0.35):.3g} eV).",
        "- Ion force sign check: the one-dimensional closure gives Vix*dVix/dx + dphi/dx = 0, "
        "so the conservative residual uses +ni*dphi/dx.",
        f"- nu_ei final mean = {final_loss.get('nu_ei_mean_Hz', float('nan')):.6e} Hz.",
        f"- nu_en final mean = {final_loss.get('nu_en_mean_Hz', float('nan')):.6e} Hz.",
        f"- final mean |ion friction/loading/e-drag x| = {final_loss.get('ion_friction_x_abs_mean', float('nan')):.6e}; "
        f"mean |ni*dphi/dx| = {final_loss.get('ion_electric_x_abs_mean', float('nan')):.6e}; "
        f"friction/electric = {final_loss.get('ion_friction_over_electric_x', float('nan')):.6g}.",
        f"- Hall momentum weights: x={weights.get('elec_mom_x', 0.0):.3g}, "
        f"r={weights.get('elec_mom_r', 0.0):.3g}, theta={weights.get('elec_mom_theta', 0.0):.3g}.",
        f"- Data weights: ni={data_weights['ni']:.3g}, Te={data_weights['Te']:.3g}, "
        f"Vp={data_weights['Vp']:.3g}.",
        "- Data ni loss = log-density MSE; Te and Vp data losses = normalized MSE.",
        f"- Ion continuity weight = {weights.get('ion_cont', float('nan')):.6g}.",
        f"- Ion momentum weights: x={weights.get('ion_mom_x', 0.0):.3g}, "
        f"r={weights.get('ion_mom_r', 0.0):.3g}.",
        f"- Direct-ni hard anode target = {physics['ni_inlet']:.6g} normalized "
        f"= {physics['ni_inlet'] * physics['n_ref']:.6e} m^-3.",
        f"- Direct-ni HG-amplitude seed peak = {physics.get('ni_seed_peak', float('nan')):.6g} "
        f"normalized; seed amp = {physics.get('ni_seed_amp', float('nan')):.6g}.",
        f"- Vix relaxed anode target = {physics.get('Vix_anode_min', float('nan')):.6g} normalized "
        f"= {physics.get('Vix_anode_min', float('nan')) * physics['V_ref']:.6g} m/s "
        f"(Bohm-criterion speed {physics['Vix_inlet'] * physics['V_ref']:.6g} m/s).",
        f"- Neutral axial bulk speed Vn = {physics['Vn'] * physics['V_ref']:.6g} m/s "
        f"({physics['Vn']:.6g} normalized).",
        "",
        "Guessed constants flagged for review:",
        "",
        f"- energy_alpha = {physics.get('energy_alpha', 1.0):.3g} (excitation/ionization multiplier).",
        f"- sigma_en = {physics.get('sigma_en', 2.5e-19):.3e} m^2 for Xe.",
        f"- Coulomb log = {physics.get('coulomb_log', 10.0):.3g}.",
        f"- Tn = {physics.get('Tn_eV', 0.025):.3g} eV and Ti = {physics.get('Ti_eV', 0.025):.3g} eV.",
        "",
        "## Before/after losses",
        "",
        "| Term | init | final |",
        "|---|---:|---:|",
    ]
    for key in [
        "total", "poisson", "ion_cont", "ion_cont_raw", "neut_cont", "elec_cont", "energy",
        "ion_mom_x", "ion_mom_r", "elec_mom_x", "elec_mom_r", "elec_mom_theta",
        "JT_target", "JT_current_A", "nn_exit", "smooth_ni", "smooth_nn",
        "smooth_phi", "smooth_Er", "smooth_Te",
    ]:
        lines.append(
            f"| {key} | {init_loss.get(key, float('nan')):.6g} | "
            f"{final_loss.get(key, float('nan')):.6g} |"
        )

    lines += [
        "",
        "## Field metrics",
        "",
        f"- ni_peak_value = {s['ni_peak_value']:.8g}; ni_peak_x = {s['ni_peak_x']:.8g}.",
        f"- Te_peak_value = {s['Te_peak_value_norm']:.8g} normalized = "
        f"{s['Te_peak_value_eV']:.8g} eV; Te_peak_x = {s['Te_peak_x']:.8g}.",
        f"- Ex_peak_x = {s['Ex_peak_x']:.8g}.",
        f"- nn(x=0.55, centerline) = {s['nn_x055_centerline']:.8g}.",
        f"- RMS (200x80): IonCont raw = {s['IonCont_RMS']:.8g}; "
        f"IonCont scaled = {s.get('IonCont_scaled_RMS', float('nan')):.8g}; "
        f"NeutCont = {s['NeutCont_RMS']:.8g}; Poisson = {s['Poisson_RMS']:.8g}.",
        f"- eta min/max/mean = {s['eta_stats']['min']:.8g} / "
        f"{s['eta_stats']['max']:.8g} / {s['eta_stats']['mean']:.8g}.",
        f"- eta sat95 fraction = {s['eta_stats']['frac_gt_0.95_eta_max']:.8g}.",
        f"- JT residual RMS = {s['JT_residual_RMS']:.8e}.",
        f"- JT trained = {physics.get('JT_trained', float('nan')):.8g} normalized "
        f"= {physics.get('JT_trained', float('nan')) * physics.get('JT_A_scale', float('nan')):.8g} A "
        f"(target {physics.get('JT_target_A', float('nan')):.8g} A).",
    ]

    if data_initial is not None and data_final is not None:
        lines += [
            "",
            "## Data loss",
            "",
            "| Quantity | init MSE | final MSE | init L2 rel | final L2 rel |",
            "|---|---:|---:|---:|---:|",
        ]
        for q in ("ni", "Te", "Vp"):
            lines.append(
                f"| {q} | {data_initial['losses'].get(q, float('nan')):.6g} | "
                f"{data_final['losses'].get(q, float('nan')):.6g} | "
                f"{data_initial['l2'].get(q, float('nan')):.6g} | "
                f"{data_final['l2'].get(q, float('nan')):.6g} |"
            )

    lines += [
        "",
        "## Training trajectory",
        "",
        "| epoch | total_loss | ni_peak | ni_peak_x | sat_frac_95 | IonCont_RMS | NeutCont_RMS | Poisson_RMS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in trajectory:
        lines.append(
            f"| {t['epoch']} | {t['total_loss']:.5g} | {t['ni_peak_value']:.5g} | "
            f"{t['ni_peak_x']:.4g} | {t['sat_frac_95']:.4g} | "
            f"{t['IonCont_RMS']:.5g} | {t['NeutCont_RMS']:.5g} | {t['Poisson_RMS']:.5g} |"
        )

    lines += ["", "## Stop diagnosis", ""]
    if early_stop:
        last = trajectory[-1] if trajectory else None
        lines.append(f"Stopped early at epoch {final_epoch}.")
        lines.append(f"Reason: {stop_reason}")
        if last:
            lines.append("")
            lines.append("Last monitor snapshot:")
            lines.append(f"- ni_peak = {last['ni_peak_value']:.4g} at x = {last['ni_peak_x']:.4g}")
            lines.append(f"- sat_frac_95 = {last['sat_frac_95']:.4g} (threshold 0.85)")
            lines.append(f"- sat_frac_50 = {last['sat_frac_50']:.4g}")
            lines.append(f"- eta range: [{last['eta_min']:.4g}, {last['eta_max_val']:.4g}], mean = {last['eta_mean']:.4g}")
            lines.append(f"- IonCont_RMS = {last['IonCont_RMS']:.4g}")
            lines.append(f"- NeutCont_RMS = {last['NeutCont_RMS']:.4g}")
            lines.append(f"- Poisson_RMS = {last['Poisson_RMS']:.4g}")
    else:
        lines.append("Completed the requested validation epochs without triggering an early-stop rule.")

    with open(SUMMARY_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {SUMMARY_OUT}", flush=True)

if __name__ == "__main__":
    main()
