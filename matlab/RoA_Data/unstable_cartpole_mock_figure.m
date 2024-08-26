%% Figure Generation
clear; clc; close all;

load('cartpole_unstable_roa.mat')
load('cartpole_roa_ppcvttq1_fbl.mat')
roa_fbl = roa_ppcvttq1; clear roa_ppcvttq1;
load('cartpole_roa_ppcvttq1_pd.mat')
roa_pd = roa_ppcvttq1; clear roa_ppcvttq1;
load('cartpole_roa_ppcvttq1_pd_ff.mat')
roa_pd_ff = roa_ppcvttq1; clear roa_ppcvttq1;

%% Plot
figure(1)
hold on
% Learned PD
learned = scatter(roa_pd_ff(:, 1), roa_pd_ff(:, 2), 'DisplayName', 'Learned ZDP (PD)');
learned.MarkerFaceColor = "#000000";
learned.MarkerEdgeColor = "#000000";

% Learned FBL
learned = scatter(roa_fbl(:, 1), roa_fbl(:, 2), 'DisplayName', 'Learned ZDP (FBL)');
learned.MarkerFaceColor = "#7E2F8E";
learned.MarkerEdgeColor = "#7E2F8E";

% LQR Surf
sths = ths_unstable;
sths(surf_unstable == 0) = NaN;
sdths = dths_unstable;
sdths(surf_unstable == 0) = NaN;
lqr_surf = scatter(sths(:), sdths(:), 'DisplayName', 'LQR ZDP');
lqr_surf.MarkerFaceColor = "#A2142F";
lqr_surf.MarkerEdgeColor = "#A2142F";

% LQR
lths = ths_unstable;
lths(lqr_unstable == 0) = NaN;
ldths = dths_unstable;
ldths(lqr_unstable == 0) = NaN;
lqr = scatter(lths(:), ldths(:), 'DisplayName', 'LQR');
lqr.MarkerFaceColor = "#0072BD";
lqr.MarkerEdgeColor = "#0072BD";

xlabel('$\theta$', 'Interpreter','latex')
ylabel('$\dot{\theta}$', 'Interpreter','latex')
legend()
ylim([-7, 7])