function [t, x] = simulate_ode(x0, dyn, tspan, odeopts, clip)
if nargin > 4
    [t, x] = ode45(@(t, x) simulate_dynamics(x, simulate_zd_fbl(x, dyn, clip), dyn), tspan, x0, odeopts);
else
    [t, x] = ode45(@(t, x) simulate_dynamics(x, simulate_zd_fbl(x, dyn), dyn), tspan, x0, odeopts);
end
end