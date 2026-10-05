import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_output_folders(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Big_Plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Individual_Plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Loss_Plots"), exist_ok=True)

def sample_domain_points(n_points, domain, device):
    x = (
        torch.rand(n_points, device=device, dtype=torch.float64)
        * (domain["x_max"] - domain["x_min"])
        + domain["x_min"]
    )
    r = (
        torch.rand(n_points, device=device, dtype=torch.float64)
        * (domain["r_max"] - domain["r_min"])
        + domain["r_min"]
    )
    x.requires_grad_(True)
    r.requires_grad_(True)
    return x, r

def sample_boundary_points(n_inlet, n_wall, domain, device):
    r_inlet = (
        torch.rand(n_inlet, device=device, dtype=torch.float64)
        * (domain["r_max"] - domain["r_min"])
        + domain["r_min"]
    )
    x_inlet = torch.full_like(r_inlet, domain["x_min"])

    x_wall = (
        torch.rand(2 * n_wall, device=device, dtype=torch.float64)
        * (domain["x_max"] - domain["x_min"])
        + domain["x_min"]
    )
    r_wall = torch.cat([
        torch.full((n_wall,), domain["r_min"], device=device, dtype=torch.float64),
        torch.full((n_wall,), domain["r_max"], device=device, dtype=torch.float64),
    ])

    return {
        "x_inlet": x_inlet,
        "r_inlet": r_inlet,
        "x_wall":  x_wall,
        "r_wall":  r_wall,
    }

def _grad(u, var):
    return torch.autograd.grad(
        u.sum(), var, create_graph=True, retain_graph=True
    )[0]

def compute_Br(x, physics):
    return physics["B_max"] * torch.exp(
        -((x - physics["x_B"]) ** 2) / (physics["sigma_B"] ** 2)
    )

def compute_ki(Te_norm, physics):
    Te_eV    = Te_norm * physics["phi_ref"]
    Te_floor = physics["ki_floor_eV"]
    beta     = physics["ki_softplus_beta"]
    Te_eff   = Te_floor + F.softplus(beta * (Te_eV - Te_floor)) / beta

    ki_phys = (
        1.9435e-5 * Te_eff**3
        - 0.0068  * Te_eff**2
        + 0.6705  * Te_eff
        - 1.6329
    ) * 1e-14

    ki_phys = torch.clamp(ki_phys, min=0.0)

    taper_center = physics.get("ki_taper_center_eV", 5.0)
    taper_width = physics.get("ki_taper_width_eV", 0.35)
    taper = torch.sigmoid((Te_eff - taper_center) / taper_width)
    ki_phys = ki_phys * taper

    ki_pos = torch.clamp(ki_phys, min=1e-30)
    return physics["ki_scale"] * ki_pos / physics["k_ref"]

def compute_Si(nn_, ne, Te_norm, physics):
    ki_hat = compute_ki(Te_norm, physics)
    return ki_hat * nn_ * ne

def compute_collision_frequencies(ne, nn_, Te_norm, physics):
    """Return electron collision frequencies in Hz and normalized forms.

    nu_hat_momentum uses the legacy v26 Hall-momentum scale nu_ref.
    nu_hat_energy uses L/V_ref because the energy equation is written against
    dimensionless spatial convection terms.
    """
    Te_eV = torch.clamp(Te_norm * physics["phi_ref"], min=0.05)
    ne_phys = torch.clamp(ne, min=1e-12) * physics["n_ref"]
    nn_phys = torch.clamp(nn_, min=1e-12) * physics["n_ref"]

    ne_cm3 = ne_phys / 1.0e6
    coulomb_log = physics.get("coulomb_log", 10.0)
    nu_ei = 2.91e-6 * ne_cm3 * coulomb_log / torch.clamp(Te_eV, min=0.05).pow(1.5)

    sigma_en = physics.get("sigma_en", 2.5e-19)
    e_phys = physics["e_phys"]
    me = physics["me"]
    vth_e = torch.sqrt(8.0 * Te_eV * e_phys / (np.pi * me))
    nu_en = nn_phys * sigma_en * vth_e

    nu_total = nu_ei + nu_en
    nu_hat_momentum = nu_total / physics["nu_ref"]
    nu_hat_energy = nu_total * physics.get("tau_ref", physics["L_canal"] / physics["V_ref"])
    return {
        "nu_ei": nu_ei,
        "nu_en": nu_en,
        "nu_total": nu_total,
        "nu_hat_momentum": nu_hat_momentum,
        "nu_hat_energy": nu_hat_energy,
    }

def compute_equation_residuals(model, x, r, physics):
    """Compute the shared interior physics residuals without reducing them.

    Training and validation intentionally differ only after this function:
    training reduces these tensors on its sampled collocation points and applies
    loss weights, while validation reduces them without weights on its fixed grid.
    """
    phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta = model(x, r)
    JT = model.current_density()

    Si = compute_Si(nn_, ne, Te, physics)
    coll = compute_collision_frequencies(ne, nn_, Te, physics)

    phi_x = _grad(phi, x)
    phi_r = _grad(phi, r)
    phi_xx = _grad(phi_x, x)
    phi_rr = _grad(phi_r, r)

    ni_r = _grad(ni, r)
    niVix_x = _grad(ni * Vix, x)
    niVir_r = _grad(ni * Vir, r)

    neVex_x = _grad(ne * Vex, x)
    neVer_r = _grad(ne * Ver, r)
    energy_flux_x = _grad(ne * Vex * 2.5 * Te, x)
    energy_flux_r = _grad(ne * Ver * 2.5 * Te, r)

    laplace_phi = phi_xx + phi_rr + phi_r / r
    res_poisson = (laplace_phi / physics["alpha"] - ni * eta) / model.eta_max

    res_ion_cont_raw = niVix_x + niVir_r + ni * Vir / r - Si
    ion_cont_scale = physics.get("ion_cont_scale", 1.0)
    res_ion_cont = res_ion_cont_raw / ion_cont_scale

    nn_x = _grad(nn_, x)
    res_neut_cont = physics["Vn"] * nn_x + Si
    res_elec_cont = neVex_x + neVer_r + ne * Ver / r - Si

    Ti = torch.full_like(Te, physics.get("Ti_eV", 0.025) / physics["phi_ref"])
    Tn = torch.full_like(Te, physics.get("Tn_eV", 0.025) / physics["phi_ref"])
    alpha_E = physics.get("energy_alpha", 1.0)
    eps_i = physics.get("eps_i", 1.0)
    tau_ref = physics.get("tau_ref", physics["L_canal"] / physics["V_ref"])
    nu_ei_E = coll["nu_ei"] * tau_ref
    nu_en_E = coll["nu_en"] * tau_ref
    energy_rhs = (
        3.0 * (physics["me"] / physics["mi"]) * ne * nu_ei_E * (Ti - Te)
        + 3.0 * (physics["me"] / physics["mn"]) * ne * nu_en_E * (Tn - Te)
        + Si * (1.5 * Te + alpha_E * eps_i)
    )
    res_energy_raw = (
        energy_flux_x + energy_flux_r + ne * Ver * 2.5 * Te / r
        - ne * Vex * phi_x - ne * Ver * phi_r
        - energy_rhs
    )
    energy_scale = physics.get("energy_scale", 1.0)
    res_energy = res_energy_raw / energy_scale

    niVix2_x = _grad(ni * Vix * Vix, x)
    niVixVir_x = _grad(ni * Vix * Vir, x)
    niVixVir_r = _grad(ni * Vix * Vir, r)
    niVir2_r = _grad(ni * Vir * Vir, r)
    ion_electric_x = ni * phi_x
    ion_electric_r = ni * phi_r
    nu_in_hat = physics.get("nu_in", 1.8e6) * tau_ref
    ion_neutral_factor = (physics["mn"] / physics["mi"]) * nu_in_hat
    electron_drag_force_x = ni * (physics["me"] / physics["mi"]) * coll["nu_total"] * tau_ref * (Vex - Vix)
    electron_drag_force_r = ni * (physics["me"] / physics["mi"]) * coll["nu_total"] * tau_ref * (Ver - Vir)
    electron_drag_residual_x = -electron_drag_force_x
    electron_drag_residual_r = -electron_drag_force_r
    ion_neutral_friction_x = ni * ion_neutral_factor * (Vix - physics["Vn"])
    ionization_momentum_x = -Si * physics["Vn"]
    ion_neutral_friction_r = ni * ion_neutral_factor * Vir
    ionization_momentum_r = torch.zeros_like(Vir)
    res_ion_mom_x_raw = (
        niVix2_x + niVixVir_r + ni * Vix * Vir / r
        + ion_electric_x
        + electron_drag_residual_x
        + ion_neutral_friction_x
        + ionization_momentum_x
    )
    res_ion_mom_r_raw = (
        niVixVir_x + niVir2_r + ni * Vir * Vir / r
        + ion_electric_r
        + electron_drag_residual_r
        + ion_neutral_friction_r
        + ionization_momentum_r
    )
    ion_mom_scale = physics.get("ion_mom_scale", 1.0)
    res_ion_mom_x = res_ion_mom_x_raw / ion_mom_scale
    res_ion_mom_r = res_ion_mom_r_raw / ion_mom_scale

    res_current = ni * Vix - ne * Vex - JT

    Br = compute_Br(x, physics)
    momentum_mass_ratio = physics["me"] / physics["mi"]
    momentum_tau_ref = physics.get("tau_ref", physics["L_canal"] / physics["V_ref"])
    omega_ce = physics["e_over_me"] * Br
    nu_momentum = coll["nu_total"] + physics["alpha_B"] * omega_ce
    omega_ce_hat = momentum_mass_ratio * omega_ce * momentum_tau_ref
    nu_hat_momentum_consistent = momentum_mass_ratio * nu_momentum * momentum_tau_ref
    Vex_x = _grad(Vex, x)
    Vex_r = _grad(Vex, r)
    Ver_x = _grad(Ver, x)
    Ver_r = _grad(Ver, r)
    Vetheta_x = _grad(Vetheta, x)
    Vetheta_r = _grad(Vetheta, r)
    res_elec_mom_x = (
        momentum_mass_ratio * (ne * Vex * Vex_x + ne * Ver * Vex_r)
        - ne * phi_x
        - omega_ce_hat * ne * Vetheta
        + ne * nu_hat_momentum_consistent * Vex
    )
    res_elec_mom_r = (
        momentum_mass_ratio * (ne * Vex * Ver_x + ne * Ver * Ver_r)
        - ne * phi_r
        + ne * nu_hat_momentum_consistent * Ver
    )
    res_elec_mom_theta = (
        momentum_mass_ratio * (ne * Vex * Vetheta_x + ne * Ver * Vetheta_r)
        + omega_ce_hat * ne * Vex
        + ne * nu_hat_momentum_consistent * Vetheta
    )

    return {
        "fields": (phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta),
        "JT": JT,
        "poisson": res_poisson,
        "ion_cont_raw": res_ion_cont_raw,
        "ion_cont": res_ion_cont,
        "neut_cont": res_neut_cont,
        "elec_cont": res_elec_cont,
        "energy": res_energy,
        "ion_mom_x": res_ion_mom_x,
        "ion_mom_r": res_ion_mom_r,
        "current": res_current,
        "elec_mom_x": res_elec_mom_x,
        "elec_mom_r": res_elec_mom_r,
        "elec_mom_theta": res_elec_mom_theta,
        "Si": Si,
        "coll": coll,
        "phi_rr": phi_rr,
        "ni_r": ni_r,
        "ion_cont_scale": ion_cont_scale,
        "energy_scale": energy_scale,
        "ion_mom_scale": ion_mom_scale,
        "ion_electric_x": ion_electric_x,
        "nu_in_hat": nu_in_hat,
        "ion_neutral_friction_x": ion_neutral_friction_x,
        "ionization_momentum_x": ionization_momentum_x,
        "electron_drag_residual_x": electron_drag_residual_x,
        "omega_ce_hat": omega_ce_hat,
    }

def compute_all_losses(model, x, r, boundary_data, domain, physics, weights):
    """
    Loss function for the 2D cylindrical HET PINN with AMBIPOLAR Poisson.

    Equations solved:
      - Poisson (ambipolar): ∇²φ̂ - α*n̂i*η = 0
      - Ion continuity:      ∂x(n̂i V̂ix) + ∂r(n̂i V̂ir) + n̂i V̂ir/r - Ŝi = 0
      - Neutral continuity:  ∂x(V̂n n̂n) + Ŝi = 0   [flux form, V̂n constant]
      - Electron continuity: ∂x(n̂e V̂ex) + ∂r(n̂e V̂er) + n̂e V̂er/r - Ŝi = 0
      - Ion momentum:      conservative 2D momentum with distributed birth source
      - BCs: nn_inlet, Vetheta_inlet, Vir/Ver walls, JT_target, nn_exit_target

    ne = ni*(1+eta) enforced in model.forward.
    Vix is a solved positive network output with hard anode inlet.
    """
    equation = compute_equation_residuals(model, x, r, physics)
    phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta = equation["fields"]
    JT = equation["JT"]
    res_poisson = equation["poisson"]
    res_ion_cont_raw = equation["ion_cont_raw"]
    res_ion_cont = equation["ion_cont"]
    res_neut_cont = equation["neut_cont"]
    res_elec_cont = equation["elec_cont"]
    res_energy = equation["energy"]
    res_ion_mom_x = equation["ion_mom_x"]
    res_ion_mom_r = equation["ion_mom_r"]
    res_current = equation["current"]
    res_elec_mom_x = equation["elec_mom_x"]
    res_elec_mom_r = equation["elec_mom_r"]
    res_elec_mom_theta = equation["elec_mom_theta"]
    Si = equation["Si"]
    coll = equation["coll"]
    phi_rr = equation["phi_rr"]
    ni_r = equation["ni_r"]
    ion_cont_scale = equation["ion_cont_scale"]
    energy_scale = equation["energy_scale"]
    ion_mom_scale = equation["ion_mom_scale"]
    ion_electric_x = equation["ion_electric_x"]
    nu_in_hat = equation["nu_in_hat"]
    ion_neutral_friction_x = equation["ion_neutral_friction_x"]
    ionization_momentum_x = equation["ionization_momentum_x"]
    electron_drag_residual_x = equation["electron_drag_residual_x"]

    loss_poisson   = (res_poisson   ** 2).mean()
    loss_ion_cont_raw = (res_ion_cont_raw ** 2).mean()
    loss_ion_cont  = (res_ion_cont  ** 2).mean()
    loss_neut_cont = (res_neut_cont ** 2).mean()
    loss_elec_cont = (res_elec_cont ** 2).mean()
    loss_energy    = (res_energy    ** 2).mean()
    loss_ion_mom_x = (res_ion_mom_x ** 2).mean()
    loss_ion_mom_r = (res_ion_mom_r ** 2).mean()
    loss_current   = (res_current   ** 2).mean()
    loss_elec_mom_x     = (res_elec_mom_x     ** 2).mean()
    loss_elec_mom_r     = (res_elec_mom_r     ** 2).mean()
    loss_elec_mom_theta = (res_elec_mom_theta ** 2).mean()

    x_in = boundary_data["x_inlet"]
    r_in = boundary_data["r_inlet"]
    x_w  = boundary_data["x_wall"]
    r_w  = boundary_data["r_wall"]

    _, ni_in, _, _, nn_in, _, _, _, _, Vetheta_in, _ = model(x_in, r_in)
    _, _,     _, Vir_w, _, _, _, Ver_w, _, _,      _ = model(x_w,  r_w)

    ni_inlet_val  = physics["ni_inlet"]
    ni_in_safe    = torch.clamp(ni_in, min=1e-10)
    loss_ni_inlet = (torch.log(ni_in_safe / ni_inlet_val)).pow(2).mean()
    loss_nn_inlet      = ((nn_in      - physics["nn_inlet"])     ** 2).mean()
    loss_Vetheta_inlet = ((Vetheta_in - physics["Vetheta_inlet"]) ** 2).mean()
    loss_Vir_wall = (Vir_w ** 2).mean()
    loss_Ver_wall = (Ver_w ** 2).mean()
    loss_JT_target = (JT - physics["JT_norm"]) ** 2
    JT_A = JT * physics.get("JT_A_scale", 1.0)
    JT_target_A = physics.get("JT_target_A", physics["JT_norm"] * physics.get("JT_A_scale", 1.0))
    loss_JT_current_A = (JT_A - JT_target_A) ** 2

    nn_exit_target = physics.get("nn_exit_target", -1.0)
    if nn_exit_target > 0:
        x_exit = torch.full_like(r_in, domain["x_max"])
        _, _, _, _, nn_ex, _, _, _, _, _, _ = model(x_exit, r_in)
        excess = torch.relu(nn_ex - nn_exit_target)
        loss_nn_exit = (excess ** 2).mean()
    else:
        loss_nn_exit = torch.tensor(0.0, dtype=phi.dtype)

    Te_r = _grad(Te, r)
    Te_rr = _grad(Te_r, r)
    ni_rr = _grad(ni_r, r)
    nn_r  = _grad(nn_, r)
    nn_rr = _grad(nn_r, r)
    loss_smooth_phi = (phi_rr ** 2).mean()

    loss_smooth_Er = (phi_rr ** 2).mean()
    loss_smooth_ni = (ni_rr ** 2).mean()
    loss_smooth_nn = (nn_rr ** 2).mean()
    loss_smooth_Te = (Te_rr ** 2).mean()

    total = (
          weights["poisson"]       * loss_poisson
        + weights["ion_cont"]      * loss_ion_cont
        + weights["neut_cont"]     * loss_neut_cont
        + weights["elec_cont"]     * loss_elec_cont
        + weights.get("energy",    0.0) * loss_energy
        + weights["ion_mom_x"]     * loss_ion_mom_x
        + weights.get("ion_mom_r", 0.0) * loss_ion_mom_r
        + weights["ni_inlet"]      * loss_ni_inlet
        + weights["nn_inlet"]      * loss_nn_inlet
        + weights["Vetheta_inlet"] * loss_Vetheta_inlet
        + weights["Vir_wall"]      * loss_Vir_wall
        + weights["Ver_wall"]      * loss_Ver_wall
        + weights["JT_target"]     * loss_JT_target
        + weights.get("JT_current_A", 0.0) * loss_JT_current_A
        + weights.get("nn_exit",   0.0) * loss_nn_exit
        + weights.get("smooth_ni", 0.0) * loss_smooth_ni
        + weights.get("smooth_nn", 0.0) * loss_smooth_nn
        + weights.get("smooth_phi", 0.0) * loss_smooth_phi
        + weights.get("smooth_Er", 0.0) * loss_smooth_Er
        + weights.get("smooth_Te", 0.0) * loss_smooth_Te
        + weights.get("elec_mom_x",     0.0) * loss_elec_mom_x
        + weights.get("elec_mom_r",     0.0) * loss_elec_mom_r
        + weights.get("elec_mom_theta", 0.0) * loss_elec_mom_theta
        + weights.get("current",        0.0) * loss_current
    )

    loss_dict = {
        "total":          total.item(),
        "poisson":        loss_poisson.item(),
        "ion_cont":       loss_ion_cont.item(),
        "ion_cont_raw":   loss_ion_cont_raw.item(),
        "ion_cont_raw_RMS": math.sqrt(max(loss_ion_cont_raw.item(), 0.0)),
        "neut_cont":      loss_neut_cont.item(),
        "elec_cont":      loss_elec_cont.item(),
        "energy":         loss_energy.item(),
        "ion_mom_x":      loss_ion_mom_x.item(),
        "ion_mom_r":      loss_ion_mom_r.item(),
        "current":        loss_current.item(),
        "elec_mom_x":     loss_elec_mom_x.item(),
        "elec_mom_r":     loss_elec_mom_r.item(),
        "elec_mom_theta": loss_elec_mom_theta.item(),
        "ni_inlet":       loss_ni_inlet.item(),
        "nn_inlet":       loss_nn_inlet.item(),
        "Vetheta_inlet":  loss_Vetheta_inlet.item(),
        "Vir_wall":       loss_Vir_wall.item(),
        "Ver_wall":       loss_Ver_wall.item(),
        "JT_target":      loss_JT_target.item(),
        "JT_current_A":   loss_JT_current_A.item(),
        "nn_exit":        loss_nn_exit.item(),
        "smooth_ni":      loss_smooth_ni.item(),
        "smooth_nn":      loss_smooth_nn.item(),
        "smooth_phi":     loss_smooth_phi.item(),
        "smooth_Er":      loss_smooth_Er.item(),
        "smooth_Te":      loss_smooth_Te.item(),
        "JT_train":       JT.item(),
        "JT_train_A":     JT_A.item(),
        "Si_mean":        Si.mean().item(),
        "nu_ei_mean_Hz":  coll["nu_ei"].detach().mean().item(),
        "nu_en_mean_Hz":  coll["nu_en"].detach().mean().item(),
        "energy_scale":   float(energy_scale),
        "ion_mom_scale":  float(ion_mom_scale),
        "ion_cont_scale": float(ion_cont_scale),
        "ion_electric_x_abs_mean": ion_electric_x.detach().abs().mean().item(),
        "nu_in_hat": float(nu_in_hat),
        "ion_friction_x_abs_mean": (
            ion_neutral_friction_x + ionization_momentum_x + electron_drag_residual_x
        ).detach().abs().mean().item(),
        "ion_neutral_friction_x_mean": ion_neutral_friction_x.detach().mean().item(),
        "ionization_momentum_x_mean": ionization_momentum_x.detach().mean().item(),
        "electron_drag_x_mean": electron_drag_residual_x.detach().mean().item(),
        "ion_friction_over_electric_x": (
            (ion_neutral_friction_x + ionization_momentum_x + electron_drag_residual_x).detach().abs().mean()
            / torch.clamp(ion_electric_x.detach().abs().mean(), min=1e-30)
        ).item(),
        "eta_abs_mean":   eta.detach().abs().mean().item(),
        "eta_abs_max":    eta.detach().abs().max().item(),
    }

    return total, loss_dict

def _init_history():
    keys = [
        "total", "poisson",
        "ion_cont", "neut_cont", "elec_cont", "ion_mom_x", "ion_mom_r",
        "current", "ni_inlet", "nn_inlet", "Vetheta_inlet",
        "Vir_wall", "Ver_wall", "JT_target", "JT_train", "Si_mean",
        "nn_exit", "smooth_ni", "smooth_nn", "smooth_phi", "smooth_Te",
        "eta_abs_mean", "eta_abs_max",
        "ni_mean", "ni_std",
    ]
    return {k: [] for k in keys}

def train_adam(
    model, x_f, r_f, boundary_data, domain, physics,
    epochs, lr, weights, print_every=500
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6
    )
    history = _init_history()

    best_loss = float("inf")
    stagnation_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        total, loss_dict = compute_all_losses(
            model, x_f, r_f, boundary_data, domain, physics, weights
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_dict["total"])

        for k in history:
            if k not in ("ni_mean", "ni_std") and k in loss_dict:
                history[k].append(loss_dict[k])

        if epoch % print_every == 0:
            with torch.no_grad():
                x_check = x_f.detach()
                r_check = r_f.detach()
                _, ni_c, _, _, _, _, _, _, _, _, _ = model(x_check, r_check)

            ni_mean = ni_c.mean().item()
            ni_std  = ni_c.std().item()
            xi_check = (x_check - domain["x_min"]) / (domain["x_max"] - domain["x_min"])
            interior_mask = (xi_check > 0.05)
            ni_int_max = ni_c[interior_mask].max().item() if interior_mask.any() else ni_mean

            history["ni_mean"].append(ni_mean)
            history["ni_std"].append(ni_std)

            flags = []
            if ni_mean < 1e-3:
                flags.append("⚠  DENSITY COLLAPSE")
            current_loss = loss_dict["total"]
            if current_loss < best_loss * 0.99:
                best_loss = current_loss
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= 3 and epoch > print_every:
                flags.append("⚠  STAGNATION")
            flag_str = "  " + " | ".join(flags) if flags else ""
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch:5d} | "
                f"Total: {loss_dict['total']:.4f} | "
                f"Poisson: {loss_dict['poisson']:.3e} | "
                f"IonCont: {loss_dict['ion_cont']:.4f} | "
                f"NeutCont: {loss_dict['neut_cont']:.4f} | "
                f"|η|: {loss_dict['eta_abs_mean']:.2e} | "
                f"JT*: {loss_dict['JT_train']:.4f} | "
                f"ni int_max={ni_int_max:.3f} | "
                f"lr={current_lr:.2e}"
                + flag_str
            )
        else:
            history["ni_mean"].append(history["ni_mean"][-1] if history["ni_mean"] else 0.0)
            history["ni_std"].append(history["ni_std"][-1]   if history["ni_std"]  else 0.0)

    return history

def compute_validation_residuals(model, domain, physics, nx=200, nr=80, device="cpu"):
    x_lin = torch.linspace(domain["x_min"], domain["x_max"], nx, device=device, dtype=torch.float64)
    r_lin = torch.linspace(domain["r_min"], domain["r_max"], nr, device=device, dtype=torch.float64)
    XX, RR = torch.meshgrid(x_lin, r_lin, indexing="ij")

    x_flat = XX.reshape(-1).clone().detach().requires_grad_(True)
    r_flat = RR.reshape(-1).clone().detach().requires_grad_(True)

    equation = compute_equation_residuals(model, x_flat, r_flat, physics)
    _, ni, _, _, _, _, Vex, _, _, _, _ = equation["fields"]
    res_ion_raw = equation["ion_cont_raw"]
    res_ion = equation["ion_cont"]
    res_neut = equation["neut_cont"]
    res_poisson = equation["poisson"]
    res_elec = equation["elec_cont"]
    res_energy = equation["energy"]
    res_ion_mom_x = equation["ion_mom_x"]
    res_ion_mom_r = equation["ion_mom_r"]
    res_emom_x = equation["elec_mom_x"]
    res_emom_r = equation["elec_mom_r"]
    res_emom_t = equation["elec_mom_theta"]
    res_current = equation["current"]
    omega_ce_hat_v = equation["omega_ce_hat"]
    ion_electric_x = equation["ion_electric_x"]
    nu_in_hat = equation["nu_in_hat"]
    ion_neutral_friction_x = equation["ion_neutral_friction_x"]
    ionization_momentum_x = equation["ionization_momentum_x"]
    electron_drag_residual_x = equation["electron_drag_residual_x"]

    ion_raw_mse = float(res_ion_raw.pow(2).mean().item())
    ion_mse     = float(res_ion.pow(2).mean().item())
    neut_mse    = float(res_neut.pow(2).mean().item())
    pois_mse    = float(res_poisson.pow(2).mean().item())
    elec_mse    = float(res_elec.pow(2).mean().item())
    energy_mse  = float(res_energy.pow(2).mean().item())
    imom_x_mse  = float(res_ion_mom_x.pow(2).mean().item())
    imom_r_mse  = float(res_ion_mom_r.pow(2).mean().item())
    emom_x_mse  = float(res_emom_x.pow(2).mean().item())
    emom_r_mse  = float(res_emom_r.pow(2).mean().item())
    emom_t_mse  = float(res_emom_t.pow(2).mean().item())
    curr_mse    = float(res_current.pow(2).mean().item())

    return {
        "IonCont_MSE":  ion_raw_mse,
        "IonCont_scaled_MSE": ion_mse,
        "NeutCont_MSE": neut_mse,
        "Poisson_MSE":  pois_mse,
        "ElecCont_MSE": elec_mse,
        "IonCont_RMS":  ion_raw_mse ** 0.5,
        "IonCont_scaled_RMS": ion_mse ** 0.5,
        "NeutCont_RMS": neut_mse ** 0.5,
        "Poisson_RMS":  pois_mse ** 0.5,
        "ElecCont_RMS": elec_mse ** 0.5,
        "Energy_MSE":       energy_mse,
        "Energy_RMS":       energy_mse ** 0.5,
        "ion_mom_x_RMS":    imom_x_mse ** 0.5,
        "ion_mom_r_RMS":    imom_r_mse ** 0.5,
        "current_RMS":      curr_mse ** 0.5,
        "ElecMomX_RMS":     emom_x_mse ** 0.5,
        "ElecMomR_RMS":     emom_r_mse ** 0.5,
        "ElecMomTheta_RMS": emom_t_mse ** 0.5,
        "omega_ce_hat_max": float(omega_ce_hat_v.detach().max().item()),
        "Vex_max":  float(Vex.detach().max().item()),
        "Vex_min":  float(Vex.detach().min().item()),
        "Vex_mean": float(Vex.detach().mean().item()),
        "ion_electric_x_abs_mean": float(ion_electric_x.detach().abs().mean().item()),
        "nu_in_hat": float(nu_in_hat),
        "ion_friction_x_abs_mean": float(
            (ion_neutral_friction_x + ionization_momentum_x + electron_drag_residual_x).detach().abs().mean().item()
        ),
        "ion_neutral_friction_x_mean": float(ion_neutral_friction_x.detach().mean().item()),
        "ionization_momentum_x_mean": float(ionization_momentum_x.detach().mean().item()),
        "electron_drag_x_mean": float(electron_drag_residual_x.detach().mean().item()),
        "ion_friction_over_electric_x": float(
            (
                (ion_neutral_friction_x + ionization_momentum_x + electron_drag_residual_x).detach().abs().mean()
                / torch.clamp(ion_electric_x.detach().abs().mean(), min=1e-30)
            ).item()
        ),
    }

def evaluate_on_grid(model, domain, physics, nx=200, nr=200, device="cpu"):
    x_lin = torch.linspace(domain["x_min"], domain["x_max"], nx, device=device, dtype=torch.float64)
    r_lin = torch.linspace(domain["r_min"], domain["r_max"], nr, device=device, dtype=torch.float64)
    XX, RR = torch.meshgrid(x_lin, r_lin, indexing="ij")

    x_flat = XX.reshape(-1).clone().detach().requires_grad_(True)
    r_flat = RR.reshape(-1).clone().detach().requires_grad_(True)

    phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta = model(x_flat, r_flat)

    ki = compute_ki(Te, physics)
    Si = compute_Si(nn_, ne, Te, physics)
    Jx = ni * Vix - ne * Vex

    phi_x = _grad(phi, x_flat)
    phi_r = _grad(phi, r_flat)
    Ex = -phi_x
    Er = -phi_r

    def to_np(t):
        return t.reshape(nx, nr).detach().cpu().numpy()

    return (
        XX.cpu().numpy(),
        RR.cpu().numpy(),
        to_np(phi),
        to_np(ni),
        to_np(Vix),
        to_np(Vir),
        to_np(nn_),
        to_np(ne),
        to_np(Vex),
        to_np(Ver),
        to_np(Te),
        to_np(Vetheta),
        to_np(Si),
        to_np(Jx),
        to_np(ki),
        to_np(Ex),
        to_np(Er),
        to_np(eta),
    )

def save_combined_plot(
    XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er, eta,
    output_dir, run_name, physics,
):
    plt.rcParams.update({"font.size": 13})

    fields = [
        (phi,     r"Potential $\hat{\phi}(x,r)$"),
        (ni,      r"Ion density $\hat{n}_i(x,r)$"),
        (nn_,     r"Neutral density $\hat{n}_n(x,r)$"),
        (ne,      r"Electron density $\hat{n}_e(x,r)$"),
        (Te,      r"Electron temperature $\hat{T}_e(x,r)$"),
        (Vix,     r"Ion axial velocity $\hat{V}_{ix}(x,r)$"),
        (Vir,     r"Ion radial velocity $\hat{V}_{ir}(x,r)$"),
        (Vex,     r"Electron axial velocity $\hat{V}_{ex}(x,r)$"),
        (Ver,     r"Electron radial velocity $\hat{V}_{er}(x,r)$"),
        (Vetheta, r"Hall drift $\hat{V}_{e\theta}(x,r)$"),
        (Si,      r"Ionization source $\hat{S}_i(x,r)$"),
        (ki,      r"Ionization rate $\hat{k}_i(\hat{T}_e)$"),
        (Ex,      r"Electric field $\hat{E}_x(x,r)$"),
        (Er,      r"Electric field $\hat{E}_r(x,r)$"),
        (eta,     r"Deviation $\eta(x,r)$"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(24, 18))
    axes = axes.flatten()

    JT_label = physics.get("JT_trained", physics["JT_norm"])
    label = (
        f"alpha={physics['alpha']:.2e}, "
        f"JT*={JT_label:.3f}, "
        f"Vn={physics['Vn']:.3f}, "
        f"delta={physics['delta']}, "
        f"B_max={physics['B_max']}"
    )

    for i, ax in enumerate(axes):
        if i < len(fields):
            data, title = fields[i]
            im = ax.pcolormesh(XX, RR, data, shading="auto", cmap="viridis")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("x̂", fontsize=11)
            ax.set_ylabel("r̂", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    fig.suptitle(f"PINN HET — {label}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(output_dir, "Big_Plots", f"{run_name}_combined.png")
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def save_individual_plots(
    XX, RR, phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, Si, Jx, ki, Ex, Er, eta,
    output_dir, run_name,
):
    plt.rcParams.update({"font.size": 13})

    fields = {
        "phi":      (phi,     r"$\hat{\phi}(x,r)$"),
        "ni":       (ni,      r"$\hat{n}_i(x,r)$"),
        "nn":       (nn_,     r"$\hat{n}_n(x,r)$"),
        "ne":       (ne,      r"$\hat{n}_e(x,r)$"),
        "Te":       (Te,      r"$\hat{T}_e(x,r)$"),
        "Vix":      (Vix,     r"$\hat{V}_{ix}(x,r)$"),
        "Vir":      (Vir,     r"$\hat{V}_{ir}(x,r)$"),
        "Vex":      (Vex,     r"$\hat{V}_{ex}(x,r)$"),
        "Ver":      (Ver,     r"$\hat{V}_{er}(x,r)$"),
        "Vetheta":  (Vetheta, r"$\hat{V}_{e\theta}(x,r)$"),
        "Si":       (Si,      r"$\hat{S}_i(x,r)$"),
        "ki":       (ki,      r"$\hat{k}_i(\hat{T}_e)$"),
        "Ex":       (Ex,      r"$\hat{E}_x(x,r)$"),
        "Er":       (Er,      r"$\hat{E}_r(x,r)$"),
        "eta":      (eta,     r"$\eta(x,r)$ (deviation)"),
    }

    for name, (data, title) in fields.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(XX, RR, data, shading="auto", cmap="viridis")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x̂", fontsize=12)
        ax.set_ylabel("r̂", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)

        path = os.path.join(output_dir, "Individual_Plots", f"{run_name}_{name}.png")
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()

def save_loss_plot(history, output_dir, run_name, physics):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].semilogy(history["total"], color="black", lw=2)
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(history["poisson"],   label="poisson")
    axes[1].semilogy(history["ion_cont"],  label="ion_cont")
    axes[1].semilogy(history["neut_cont"], label="neut_cont")
    axes[1].semilogy(history["elec_cont"], label="elec_cont")
    axes[1].semilogy(history["JT_target"], label="JT_target")
    axes[1].set_title("Main losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].plot(history["JT_train"], label="JT trained")
    axes[2].axhline(
        y=physics["JT_norm"], color="red", linestyle="--", alpha=0.6, label="JT_norm"
    )
    axes[2].set_title(r"Trainable current $\hat{J}_T^*$")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(r"$\hat{J}_T^*$")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "Loss_Plots", f"{run_name}_Loss_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def save_global_metrics(model, domain, physics, output_dir, run_name,
                        nr=200, device="cpu"):
    L_canal = physics["L_canal"]
    n_ref   = physics["n_ref"]
    V_ref   = physics["V_ref"]
    mi      = 2.180e-25
    e_phys  = 1.602e-19

    r_lin = torch.linspace(domain["r_min"], domain["r_max"], nr,
                           device=device, dtype=torch.float64)
    r_phys = r_lin.cpu().numpy() * L_canal
    dr_phys = r_phys[1] - r_phys[0]

    x_exit = torch.ones(nr, device=device, dtype=torch.float64)
    with torch.no_grad():
        _, ni_e, Vix_e, _, nn_e, _, Vex_e, _, _, _, _ = model(x_exit, r_lin)
    ni_e  = ni_e.cpu().numpy();  Vix_e = Vix_e.cpu().numpy()
    Vex_e = Vex_e.cpu().numpy(); nn_e  = nn_e.cpu().numpy()

    x_in = torch.zeros(nr, device=device, dtype=torch.float64)
    with torch.no_grad():
        _, _, _, _, nn_in, _, _, _, _, _, _ = model(x_in, r_lin)
    nn_in = nn_in.cpu().numpy()

    Id_hat = float(np.trapz(ni_e * (Vix_e - Vex_e) * r_phys, dx=dr_phys))
    Id_A   = 2.0 * np.pi * e_phys * n_ref * V_ref * Id_hat

    nn_exit_flux  = float(np.trapz(nn_e  * r_phys, dx=dr_phys))
    nn_inlet_flux = float(np.trapz(nn_in * r_phys, dx=dr_phys))
    eta_i = 1.0 - nn_exit_flux / nn_inlet_flux if nn_inlet_flux > 0 else 0.0

    Vix_exit_phys = float(np.mean(Vix_e)) * V_ref
    Vcls = float(np.sqrt(2.0 * e_phys * 300.0 / mi))

    mdot_P5_kg = 10.46e-6
    T_beam_N   = (mi / e_phys) * Id_A * Vix_exit_phys * eta_i
    g0 = 9.81
    Isp_beam_s = T_beam_N / (mdot_P5_kg * g0)

    metrics = {
        "Discharge current [A]":     Id_A,
        "Ionization efficiency [—]": eta_i,
        "Mean Vix(exit) [km/s]":     Vix_exit_phys * 1e-3,
        "Vix(exit)/V_classical [—]": Vix_exit_phys / Vcls,
        "Beam Isp estimate [s]":     Isp_beam_s,
        "JT_norm (target)":          physics["JT_norm"],
        "JT_trained":                physics.get("JT_trained", float("nan")),
    }

    P5_ref = {
        "Discharge current [A]":     5.4,
        "Ionization efficiency [—]": 0.90,
        "Beam Isp estimate [s]":     1640.0,
    }

    lines = [
        f"{'='*72}",
        f"  Performance Metrics — {run_name}",
        f"{'='*72}",
        f"  {'Quantity':<40} {'PINN':>10} {'P5 ref':>14}",
        f"  {'-'*67}",
    ]
    for key, val in metrics.items():
        ref = P5_ref.get(key)
        ref_str = f"{ref:>14.2f}" if ref is not None else f"{'—':>14}"
        lines.append(f"  {key:<40} {val:>10.3f} {ref_str}")
    lines.append(f"{'='*72}")
    report = "\n".join(lines)
    print(report)

    txt_path = os.path.join(output_dir, f"{run_name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved: {txt_path}")
    return metrics

DATA_OUTPUT_INDICES = {"ni": 1, "Te": 8, "Vp": 0}

_DATA_LOSS_MAP_PRINTED = False

def load_haas_gallimore_data(path, domain, physics, device, dtype=torch.float64,
                             include_r_mm=None):
    """Load Haas/Gallimore P5 experimental data, normalize to network units.

    CSV columns: x_mm, r_mm, quantity, value, unit. x_mm is measured from the
    anode; r_mm is measured from the inner wall. Points outside the simulation
    domain in normalized coordinates are dropped (printed for sanity).

    Normalization (mirrors the rest of the code):
      x_n = x_mm * 1e-3 / L_canal
      r_n = (r_in_phys + r_mm * 1e-3) / L_canal      (r_in_phys = r_min * L_canal)
      ni  -> ni / n_ref           (m^-3 -> normalized)
      Te  -> Te / phi_ref         (eV   -> normalized; phi_ref = Ei_eV = 12.1)
      Vp  -> Vp / phi_ref         (V    -> normalized phi; phi_right = 0 = cathode ref)

    Returns dict { 'ni': {'coords': (N,2), 'values': (N,)}, 'Te': ..., 'Vp': ... }.
    """
    L_canal   = physics["L_canal"]
    n_ref     = physics["n_ref"]
    phi_ref   = physics["phi_ref"]
    r_in_phys = domain["r_min"] * L_canal

    raw_data = np.genfromtxt(
        path, delimiter=",", names=True,
        dtype=[("x_mm", "f8"), ("r_mm", "f8"), ("quantity", "U8"),
               ("value", "f8"), ("unit", "U8")],
        encoding="utf-8",
    )
    if include_r_mm is None:
        include_r_values = None
    else:
        include_r_values = np.atleast_1d(include_r_mm).astype(float)

    buckets = {"ni": [], "Te": [], "Vp": []}
    dropped = {"ni": 0, "Te": 0, "Vp": 0}
    filtered = {"ni": 0, "Te": 0, "Vp": 0}
    for row in raw_data:
        q = str(row["quantity"])
        if q not in buckets:
            continue
        if include_r_values is not None and not np.any(
            np.isclose(float(row["r_mm"]), include_r_values, atol=1e-9)
        ):
            filtered[q] += 1
            continue
        x_n = float(row["x_mm"]) * 1.0e-3 / L_canal
        r_n = (r_in_phys + float(row["r_mm"]) * 1.0e-3) / L_canal
        if not (domain["x_min"] <= x_n <= domain["x_max"]
                and domain["r_min"] <= r_n <= domain["r_max"]):
            dropped[q] += 1
            continue
        v = float(row["value"])
        if q == "ni":
            v_n = v / n_ref
        else:
            v_n = v / phi_ref
        buckets[q].append((x_n, r_n, v_n))

    out = {}
    if include_r_values is None:
        print("Haas/Gallimore P5 (300V/5.4A) — data loaded (in-domain / dropped):")
    else:
        r_txt = ", ".join(f"{v:g}" for v in include_r_values)
        print(
            "Haas/Gallimore P5 (300V/5.4A) — data loaded "
            f"with r_mm filter [{r_txt}] (in-domain / filtered / dropped):"
        )
    for q, rows in buckets.items():
        if rows:
            arr = np.asarray(rows, dtype=np.float64)
            coords = torch.tensor(arr[:, :2], dtype=dtype, device=device)
            values = torch.tensor(arr[:, 2],  dtype=dtype, device=device)
        else:
            coords = torch.zeros((0, 2), dtype=dtype, device=device)
            values = torch.zeros((0,),   dtype=dtype, device=device)
        out[q] = {"coords": coords, "values": values}
        if include_r_values is None:
            print(f"  {q:>3s}: {len(rows):4d} kept  /  {dropped[q]:4d} outside domain")
        else:
            print(
                f"  {q:>3s}: {len(rows):4d} kept  /  {filtered[q]:4d} filtered  /  "
                f"{dropped[q]:4d} outside domain"
            )
    return out

def compute_data_loss(model, exp_data, output_indices=None):
    """Data loss against Haas/Gallimore measurements.

    ni uses log-density MSE so the hybrid loss penalizes multiplicative density
    errors and can select the measured high-density branch. Te and Vp retain
    normalized MSE. All terms are torch scalars with grad if model params have
    grad enabled, so the caller can add the weighted sum to the PDE total.
    Empty buckets contribute zero.
    """
    global _DATA_LOSS_MAP_PRINTED
    if output_indices is None:
        output_indices = DATA_OUTPUT_INDICES
    if not _DATA_LOSS_MAP_PRINTED:
        print(f"compute_data_loss output index mapping: {output_indices}")
        print("  PINN.forward returns "
              "(0=phi, 1=ni, 2=Vix, 3=Vir, 4=nn, 5=ne, 6=Vex, 7=Ver, "
              "8=Te, 9=Vetheta, 10=eta)")
        _DATA_LOSS_MAP_PRINTED = True

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    out = {}
    total = torch.zeros((), device=device, dtype=dtype)
    for q, idx in output_indices.items():
        d = exp_data.get(q) if exp_data is not None else None
        if d is None or d["coords"].shape[0] == 0:
            out[q] = torch.zeros((), device=device, dtype=dtype)
            continue
        x = d["coords"][:, 0]
        r = d["coords"][:, 1]
        outs = model(x, r)
        pred = outs[idx]
        target = d["values"]
        if q == "ni":
            eps = torch.as_tensor(1e-8, dtype=dtype, device=device)
            mse = torch.mean((torch.log(torch.clamp(pred, min=eps))
                              - torch.log(torch.clamp(target, min=eps))) ** 2)
        else:
            mse = torch.mean((pred - target) ** 2)
        out[q] = mse
        total = total + mse
    out["total"] = total
    return out

def data_loss_l2_relative(model, exp_data, output_indices=None):
    """Per-quantity L2 relative error ||pred - target|| / ||target||."""
    if output_indices is None:
        output_indices = DATA_OUTPUT_INDICES
    if exp_data is None:
        return {q: float("nan") for q in output_indices}
    out = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for q, idx in output_indices.items():
                d = exp_data.get(q)
                if d is None or d["coords"].shape[0] == 0:
                    out[q] = float("nan")
                    continue
                x = d["coords"][:, 0]
                r = d["coords"][:, 1]
                pred = model(x, r)[idx]
                target = d["values"]
                num = torch.sqrt(torch.mean((pred - target) ** 2))
                den = torch.sqrt(torch.mean(target ** 2)).clamp_min(1e-30)
                out[q] = float((num / den).item())
    finally:
        if was_training:
            model.train()
    return out
