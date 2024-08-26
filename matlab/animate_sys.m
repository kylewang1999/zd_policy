function animate_sys(t_fine, x_fine, dyn)
if strcmp(dyn.system, 'cartpole')
    % axis off
    base = rectangle('Position',[-0.5+x_fine(1,1) -0.375 1 .75],'linewidth',2,'Curvature',0.2);
    hold on;
    P1_out = line([x_fine(1,1) x_fine(1,1)+sin(x_fine(1,2))],[0 cos(x_fine(1,2))],'color','k','marker','o','markerSize',10,'linewidth',20);
    P1_in = line([x_fine(1,1) x_fine(1,1)+sin(x_fine(1,2))],[0 cos(x_fine(1,2))],'color',[0 0.4470 0.7410],'marker','o','markerSize',10,'linewidth',15);
    axis equal
    ax = gca;
    set(ax,'YLim',[-1.5 1.5])
    buff = 5;
    set(ax,'XLim',[x_fine(1,1)-buff x_fine(1,1)+buff])
elseif strcmp(dyn.system, 'acrobot')
    axis off
    base = rectangle('Position',[-0.5 -0.1 1.0 0.2],'linewidth',2,'Curvature',0.2);
    hold on;
    P2_out = line([sin(x_fine(1,1)) sin(x_fine(1,1))+sin(x_fine(1,1)+x_fine(1,2))],[cos(x_fine(1,1)) cos(x_fine(1,1))+cos(x_fine(1,1)+x_fine(1,2))],'color','k','marker','o','markerSize',10,'linewidth',20);
    P2_in = line([sin(x_fine(1,1)) sin(x_fine(1,1))+sin(x_fine(1,1)+x_fine(1,2))],[cos(x_fine(1,1)) cos(x_fine(1,1))+cos(x_fine(1,1)+x_fine(1,2))],'color',[0 0.4470 0.7410],'marker','o','markerSize',10,'linewidth',15);
    P1_out = line([0 sin(x_fine(1,1))],[0 cos(x_fine(1,1))],'color','k','marker','o','markerSize',10,'linewidth',20);
    P1_in = line([0 sin(x_fine(1,1))],[0 cos(x_fine(1,1))],'color',[0 0.4470 0.7410],'marker','o','markerSize',10,'linewidth',15);
    axis([-2 2 -2 2])
    axis equal
    ax = gca;
    set(ax,'YLim',[-2.5 2.5])
    set(ax, 'XLim', [min(x(:, 1)) - 1, max(x(:, 1)) + 1])
    buff = 5;
end
set(gca,'FontSize',17)
set(gca,'linewidth',2)

% figure(2)
% plot(t_fine, x_fine)
% legend('$x$', '$\theta$', '$\dot{x}$', '$\dot{\theta}$','interpreter','latex')
% xlabel('$t$','interpreter','latex')
% ylabel('state')
    figure(1)
    % axis([-2, 6, -1.5, 1.5])
    ii = 1;
    % M(ii) = getframe;
    pause
    tic
    cont = true;
    ind = 1;
    while cont
        ii = ii + 1;
        if contains(dyn.system, 'cartpole')
            set(base,'Position',[-0.5+x_fine(ind,1) -0.375 1 .75])
            set(P1_out,'XData',[x_fine(ind,1) x_fine(ind,1)+sin(x_fine(ind,2))])
            set(P1_out,'YData',[0 cos(x_fine(ind,2))])
            set(P1_in,'XData',[x_fine(ind,1) x_fine(ind,1)+sin(x_fine(ind,2))])
            set(P1_in,'YData',[0 cos(x_fine(ind,2))])
            set(ax,'XLim',[x_fine(ind,1)-buff x_fine(ind,1)+buff])
        elseif contains(dyn.system, 'acrobot')
            set(P1_out,'XData',[0 sin(x_fine(ind,1))])
            set(P1_out,'YData',[0 cos(x_fine(ind,1))])
            set(P1_in,'XData',[0 sin(x_fine(ind,1))])
            set(P1_in,'YData',[0 cos(x_fine(ind,1))])
            set(P2_out,'XData',[sin(x_fine(ind,1)) sin(x_fine(ind,1))+sin(x_fine(ind,1)+x_fine(ind,2))])
            set(P2_out,'YData',[cos(x_fine(ind,1)) cos(x_fine(ind,1))+cos(x_fine(ind,1)+x_fine(ind,2))])
            set(P2_in,'XData',[sin(x_fine(ind,1)) sin(x_fine(ind,1))+sin(x_fine(ind,1)+x_fine(ind,2))])
            set(P2_in,'YData',[cos(x_fine(ind,1)) cos(x_fine(ind,1))+cos(x_fine(ind,1)+x_fine(ind,2))])
        end
        % axis([-2, 6, -1.5, 1.5])
        drawnow
        % M(ii) = getframe;
        time = toc;
        ind = find(t_fine>time,1,'first');
        % ind = ind + 1;
        if isempty(ind) || ind >= size(x_fine, 1)
            cont = false;
        end
    end