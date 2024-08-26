function u = simulate_zd_fbl(x, dyn, clip)
% Gains
K_ll = dyn.K_ll;
Psi_z = dyn.Psi_z;

% Grab eta, z coordinates
n_z = dyn.Phi(x);
n = [eye(dyn.r) zeros(dyn.r, dyn.n - dyn.r)] * n_z;
z = [zeros(dyn.n-dyn.r, dyn.r) eye(dyn.n - dyn.r)] * n_z;

% Compute z dyn and derivatives
z_dot = dyn.omega(n_z);
Lf_omega = dyn.Lf_omega(n_z);
Lg_omega = dyn.Lg_omega(n_z);

% Compute feedback linearizing input
if dyn.r == 2
u = (dyn.LgLfy(x) - Psi_z * Lg_omega) \ ...
    (-(dyn.Lf2y(x) - Psi_z * Lf_omega) - K_ll * [n(1) - Psi_z * z; n(2) - Psi_z * z_dot]);
elseif dyn.r == 4
u = (dyn.LgLf3y(x) - dyn.LghatdddPsi_z(n_z)) \ ...
    (-(dyn.Lf4y(x) - dyn.LfhatdddPsi_z(n_z)) - K_ll * [n(1) - dyn.Psi_z(z); n(2) - dyn.dPsi_z(n_z); n(3) - dyn.ddPsi_z(n_z); n(4) - dyn.dddPsi_z(n_z)]);
end
if nargin > 2
    u = max(min(u, clip), -clip);
end
end