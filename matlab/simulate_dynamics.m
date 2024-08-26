function dx = simulate_dynamics(x, u, dyn)
dx = dyn.f(x) + dyn.g(x) * u;
end