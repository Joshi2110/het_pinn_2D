import math
import torch
import torch.nn as nn

VETHETA_MAX = 2000.0

class PINN(nn.Module):
    """2D cylindrical Hall thruster PINN. Ten raw channels, eleven physical fields.

    raw0 phi shape, raw1 ni log-amplitude, raw2 Vex correction, raw3 Vir,
    raw4 nn depletion, raw5 Ver, raw6 Vetheta, raw7 eta, raw8 Te, raw9 Vix.
    Returns (phi, ni, Vix, Vir, nn, ne, Vex, Ver, Te, Vetheta, eta).
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=10,
        hidden_dim=128,
        num_hidden_layers=4,
        phi_left=24.79,
        phi_right=0.0,
        x_min=0.0,
        x_max=1.0,
        r_min=1.579,
        r_max=2.237,
        delta=0.1,
        Vix_inlet=0.50,
        ni_inlet=0.14,
        nn_inlet=1.0,
        JT_init=0.03,
        JT_flex=0.3,
        Te_base=0.25,
        Te_peak=1.0,
        Te_min=0.025 / 12.1,
        Te_init=10.0 / 12.1,
        Te_anode=2.0 / 12.1,
        x0_Te=0.40,
        sigma_x_Te=0.12,
        r0_Te=1.9,
        sigma_r_Te=0.10,
        B_max=0.020,
        x_B=0.75,
        sigma_B=0.30,
        eta_max=1.0e-4,
        vex_epsilon=0.0,
        phi_drop_x0_init=0.65,
        phi_drop_sigma_init=0.20,
    ):
        super().__init__()

        self.phi_left  = phi_left
        self.phi_right = phi_right
        self.x_min     = x_min
        self.x_max     = x_max
        self.r_min     = r_min
        self.r_max     = r_max
        self.delta     = delta

        self.Vix_inlet = Vix_inlet
        self.ni_inlet  = ni_inlet
        self.nn_inlet  = nn_inlet

        self.JT_init = JT_init
        self.JT_flex = JT_flex

        self.JT_norm_init = JT_init
        ne_inlet_calc     = self.ni_inlet
        self.Vex_inlet    = (self.ni_inlet * self.Vix_inlet - self.JT_norm_init) / ne_inlet_calc
        self.vex_epsilon  = vex_epsilon

        self.Te_base    = Te_base
        self.Te_peak    = Te_peak
        self.Te_min     = Te_min
        self.Te_init    = Te_init
        self.Te_anode   = Te_anode
        self.x0_Te      = x0_Te
        self.sigma_x_Te = sigma_x_Te
        self.r0_Te      = r0_Te
        self.sigma_r_Te = sigma_r_Te

        self.B_max   = B_max
        self.x_B     = x_B
        self.sigma_B = sigma_B

        self.eta_max = eta_max
        self.Vetheta_max = VETHETA_MAX
        self.phi_shape_amp = 25.0
        self.Vix_anode_min = 0.05 * self.Vix_inlet
        self.ni_seed_peak = 0.060
        self.ni_seed_amp = math.log(max(self.ni_seed_peak / self.ni_inlet, 1.0))



        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        with torch.no_grad():



            self.net[-1].weight[1].zero_()
            self.net[-1].bias[1] = 0.0
            self.net[-1].bias[2] = 0.0
            self.net[-1].bias[4] = 2.5
            self.net[-1].bias[7] = 0.0

            self.net[-1].bias[8] = math.log(math.expm1(max(self.Te_init - self.Te_anode, 1e-6)))




            self.net[-1].weight[9].zero_()
            self.net[-1].bias[9] = math.log(math.expm1(5.25))

        self.JT_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def current_density(self):
        return self.JT_init * torch.exp(self.JT_flex * torch.tanh(self.JT_raw))

    def forward(self, x, r):
        inp = torch.stack([x, r], dim=1)
        raw = self.net(inp)

        Lx = self.x_max - self.x_min
        Lr = self.r_max - self.r_min

        xi  = (x - self.x_min) / Lx
        rho = (r - self.r_min) / Lr






        phi_base = self.phi_left + (self.phi_right - self.phi_left) * xi
        phi = phi_base + xi * (1.0 - xi) * self.phi_shape_amp * torch.tanh(raw[:, 0])




        Vix = self.Vix_anode_min + xi * torch.nn.functional.softplus(raw[:, 9])





        ni_seed = self.ni_seed_amp * 4.0 * xi * (1.0 - xi)
        ni = self.ni_inlet * torch.exp(ni_seed + xi * raw[:, 1])

        Vir = 0.005 * xi * torch.tanh(raw[:, 3])





        x_anode = torch.full_like(x, self.x_min)
        raw_anode = self.net(torch.stack([x_anode, r], dim=1))
        depletion_delta = raw[:, 4] - raw_anode[:, 4]
        depletion = torch.relu(
            torch.nn.functional.softplus(depletion_delta) - math.log(2.0)
        )
        nn_ = self.nn_inlet * torch.exp(-depletion)

        Ver = 0.008 * rho * (1.0 - rho) * torch.tanh(raw[:, 5])

        Vetheta = 4.0 * self.Vetheta_max * xi * (1.0 - xi) * torch.tanh(raw[:, 6])





        Te = self.Te_anode + xi * torch.nn.functional.softplus(raw[:, 8])


        eta = self.eta_max * torch.tanh(raw[:, 7] / 3.0)
        ne  = ni * (1.0 + eta)



        JT      = self.current_density()
        ne_safe = torch.clamp(ne, min=1e-6)
        Vex_closure = (ni * Vix - JT) / ne_safe
        Vex_correction = self.vex_epsilon * torch.tanh(raw[:, 2])
        Vex = Vex_closure + Vex_correction

        return phi, ni, Vix, Vir, nn_, ne, Vex, Ver, Te, Vetheta, eta
