%% ROA LQR
clear; clc; % close all;
dyn = dynamics('cartpole');
sim_options = odeset('Events',@(t, x)explosionEvent(t, x, 50));

%% LQR 
xstar = zeros(4, 1);
ustar = 0;
A = [dyn.Df_hat(xstar); dyn.Domega(xstar)];
B = [dyn.g_hat(xstar); zeros(2, 1)];
Q = eye(4);
R = 0.01;
tspan = [0, 15];
K = lqr(A, B, Q, R);
u_lqr = @(x) -K * dyn.Phi(x);

%% Compute ROA
th_max = 1.5;
dth_max = 10;
N = 151;
Beps = 0.01;

[ths, dths] = meshgrid(linspace(-th_max, th_max, N), linspace(-dth_max, dth_max, N));
lqr_stable = zeros(size(ths));
parfor ii = 1:numel(ths)
    x0 = [0; ths(ii); 0; dths(ii)];
    % [~, x] = ode45(@(t, x) simulate_dynamics(x, -K * x, dyn), tspan, x0, sim_options);
    [~, x] = ode45(@(t, x) simulate_dynamics(x, u_lqr(x), dyn), tspan, x0, sim_options);
    lqr_stable(ii) = min(vecnorm(x')) < Beps;
end
fprintf('Cartpole LQR RoA Volume: %0.4f\n', sum(lqr_stable, "all") / numel(lqr_stable) * th_max * dth_max)

%% LQR Surf
A = [dyn.Df_hat(xstar); dyn.Domega(xstar)];
B = [dyn.g_hat(xstar); zeros(2, 1)];

% LQR Computation
[Klqr, Plqr, ~] = lqr(A, B, Q, R);
[subspaces, eigenvals] = compute_zd_invariant_subspaces(A - B * Klqr, 2);

S1_eig = subspaces{1}; % Typically real
S2_eig = subspaces{2}; % Typically complex
S1 = S1_eig / S1_eig(3:4, :);
S2 = S2_eig / S2_eig(3:4, :);

% Then compute eta1d
psi1 = [1 0 0 0] * S1;
psi1_2 = [0 1 0 0] * S1; % eta2d, for the linear subspace

% Compute psi for real eigenvectors
psi2 = [1 0 0 0] * S2;
psi2_2 = [0 1 0 0] * S2;

ind = 1;
if ind == 1
    dyn.Psi_z = psi1;
    dyn.Psi2_z = psi1_2;
    dyn.K_ll = [20 2*sqrt(20)];
elseif ind == 2
    dyn.Psi_z = psi2;
    dyn.Psi2_z = psi2_2;
    dyn.K_ll = [20 2*sqrt(20)];
end

C = [1 0 -dyn.Psi_z];

dyn.K_ll = dyn.K_ll;
dyn.Psi_z = dyn.Psi_z;
dyn.Psi2_z = dyn.Psi2_z;

sim_options = odeset('Events',@(t, x)reldeg_explosion_Event(t, x, dyn, 1e-4, 100, 1e4));

%% Compute ROA
surf_stable = zeros(size(ths));
parfor ii = 1:numel(ths)
    x0 = [0; ths(ii); 0; dths(ii)];
    [t, x] = simulate_ode(x0, dyn, tspan, sim_options, 10000);
    surf_stable(ii) = min(vecnorm(x')) < Beps;
end
save('RoA_Data/cartpole_roa.mat', 'ths', 'dths', 'lqr_stable', 'surf_stable')
fprintf('Cartpole LQR Surf RoA Volume: %0.4f\n', sum(surf_stable, "all") / numel(surf_stable) * th_max * dth_max)

%% Visualize ROA
sths = ths;
sths(lqr_stable == 0) = NaN;
sdths = dths;
sdths(lqr_stable == 0) = NaN;
figure(1)
clf
plot(sths(:), sdths(:), 'r.', 'MarkerSize', 20)
xlabel("$\theta$", 'interpreter', 'latex')
ylabel("$\dot{\theta}$", 'interpreter', 'latex')
title("Cartpole Region of Attraction for $x=\dot{x}=0$", 'Interpreter','latex')

%% Visualize ROA
sths = ths;
sths(surf_stable == 0) = NaN;
sdths = dths;
sdths(surf_stable == 0) = NaN;
figure(1)
hold on
plot(sths(:), sdths(:), 'b.', 'MarkerSize', 20)
xlabel("$\theta$", 'interpreter', 'latex')
ylabel("$\dot{\theta}$", 'interpreter', 'latex')
legend('LQR', 'LQR Surf')

%% Animate something
% x0 = [0; -1.5; 0; -6];
% sim_options = odeset('Events',@(t, x)explosionEvent(t, x, 200));
% [t_lqr, x_lqr] = ode45(@(t, x) simulate_dynamics(x, u_lqr(x), dyn), tspan, x0, sim_options);
% min(vecnorm(x_lqr'))
% figure(4)
% clf
% plot(t_lqr, x_lqr)
% figure(3)
% clf
% animate_sys(t_lqr, x_lqr, dyn)
