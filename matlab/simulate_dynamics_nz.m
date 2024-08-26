function nzdot = simulate_dynamics_nz(nz, u, dyn)
dn = dyn.f_hat(nz) + dyn.g_hat(nz) * u;
dz = dyn.omega(nz);
nzdot = [dn; dz];
end