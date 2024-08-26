%% ROA LQR
clear; clc; close all;
dyn_un = dynamics('cartpole_unstable');

%% LQR 
xstar = zeros(4, 1);
A = [dyn_un.Df_hat(xstar); dyn_un.Domega(xstar)];
B = [dyn_un.g_hat(xstar); zeros(2, 1)];
Q = eye(4);
R = 0.01;
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
    dyn_un.Psi_z = psi1;
    dyn_un.Psi2_z = psi1_2;
    dyn_un.K_ll = [20 2*sqrt(20)];
elseif ind == 2
    dyn_un.Psi_z = psi2;
    dyn_un.Psi2_z = psi2_2;
    dyn_un.K_ll = [20 2*sqrt(20)];
end

C = [1 0 -dyn_un.Psi_z];
Kout = (C*A*B) \ (C*A*A + dyn_un.K_ll(1)*C + dyn_un.K_ll(2)*C*A);


dyn_un.K_ll = dyn_un.K_ll;
dyn_un.Psi_z = dyn_un.Psi_z;
dyn_un.Psi2_z = dyn_un.Psi2_z;

sim_options = odeset('Events',@(t, x)reldeg_explosion_Event(t, x, dyn_un, 1e-4, 100, 1e4));

%% Compute ROA
th_max_un = 1.5;
dth_max_un = 10;
N = 151;
Beps = 0.01;
tspan = [0, 15];

[ths_unstable, dths_unstable] = meshgrid(linspace(-th_max_un, th_max_un, N), linspace(-dth_max_un, dth_max_un, N));
surf_unstable = zeros(size(ths_unstable));
reldeg_unstable = zeros(size(ths_unstable));
tic;
parfor ii = 1:numel(ths_unstable)
    x0 = [0; ths_unstable(ii); 0; dths_unstable(ii)];
    [t, x] = simulate_ode(x0, dyn_un, tspan, sim_options, 10000);
    surf_unstable(ii) = min(vecnorm(x')) < Beps;
    reldeg_unstable(ii) = t(end) == tspan(end);
end
toc
fprintf('UCartpole LQR Surf RoA Volume: %0.4f\n', sum(surf_unstable, "all") / numel(surf_unstable) * th_max_un * dth_max_un)

%% Visualize ROA
sths = ths_unstable;
sths(surf_unstable == 0) = NaN;
sdths = dths_unstable;
sdths(surf_unstable == 0) = NaN;
figure(2)
clf
plot(sths(:), sdths(:), 'b.', 'MarkerSize', 20)
xlabel("$\theta$", 'interpreter', 'latex')
ylabel("$\dot{\theta}$", 'interpreter', 'latex')
title("Unstable Cartpole Region of Attraction for $x=\dot{x}=0$", 'Interpreter','latex')

%% Repeat for unstable cartpole
dyn_un = dynamics('cartpole_unstable');

%% LQR 
% A_un = dyn_un.df_dx(xstar);
% B_un = dyn_un.g(xstar);
A_un = [dyn_un.Df_hat(xstar); dyn_un.Domega(xstar)];
B_un = [dyn_un.g_hat(xstar); zeros(2, 1)];
[K, P] = lqr(A_un, B_un, Q, R);
u_lqr = @(x) - K * dyn_un.Phi(x);

%% Compute ROA
th_max_un = 1.5;
dth_max_un = 10;

[ths_unstable, dths_unstable] = meshgrid(linspace(-th_max_un, th_max_un, N), linspace(-dth_max_un, dth_max_un, N));
lqr_unstable = zeros(size(ths_unstable));

parfor ii = 1:numel(ths_unstable)
    x0 = [0; ths_unstable(ii); 0; dths_unstable(ii)];
    % [~, x] = ode45(@(t, x) simulate_dynamics(x, -K * x, dyn_un), tspan, x0, sim_options);
    [~, x] = ode45(@(t, x) simulate_dynamics(x, u_lqr(x), dyn_un), tspan, x0, sim_options);
lqr_unstable(ii) = min(vecnorm(x')) < Beps;
end
save('RoA_Data/cartpole_unstable_roa.mat', 'ths_unstable', 'dths_unstable', 'lqr_unstable', 'surf_unstable')
fprintf('UCartpole LQR RoA Volume: %0.4f\n', sum(lqr_unstable, "all") / numel(lqr_unstable) * th_max_un * dth_max_un)

%% Visualize ROA
sths = ths_unstable;
sths(lqr_unstable == 0) = NaN;
sdths = dths_unstable;
sdths(lqr_unstable == 0) = NaN;
figure(2)
hold on
plot(sths(:), sdths(:), 'r.', 'MarkerSize', 20)
xlabel("$\theta$", 'interpreter', 'latex')
ylabel("$\dot{\theta}$", 'interpreter', 'latex')
legend('LQR Surf', 'LQR')
