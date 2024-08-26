function [pos, isterminal,direction] = reldegEvent(~, x, dyn, tol)
denom = dyn.LgLfy(x) - dyn.Psi_z * dyn.Lg_omega(dyn.Phi(x));
pos = tol - abs(denom);
isterminal = 1;
direction = 1;
end