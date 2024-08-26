function [pos, isterminal,direction] = reldeg_explosion_Event(~, x, dyn, tol, pos_bound, u_bound)
u = simulate_zd_fbl(x, dyn);
denom = dyn.LgLfy(x) - dyn.Psi_z * dyn.Lg_omega(dyn.Phi(x));
pos = [norm(x)-pos_bound, norm(u)-u_bound, tol - abs(denom)];
isterminal = [1, 1, 1];
direction = [1, 1, 1];
end