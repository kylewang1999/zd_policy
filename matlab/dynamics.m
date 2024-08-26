function d = dynamics(system)
%%dynamics constructs the dynamics object for the given system
% cartpole or acrobot

d = struct();
d.system = system;

%%% Parameters
switch(system)
    case 'cartpole'
        %%% Cartpole
        l = 1;
        gr = 9.81;
        mc = 1;
        mp = 1;
        syms x th dx dth u real
        % syms mp mc gr l real
        q = [x; th];
        dq = [dx; dth];
        vars = [q; dq];
        D = [mc+mp mp*l*cos(th); mp*l*cos(th) mp*l^2];
        Dq = @(in)[mc+mp mp*l*cos(in(2, :)); mp*l*cos(in(2, :)) mp*l^2];
        H = [-mp*l*dth^2*sin(th); -mp*l*gr*sin(th)];
        B = [1; 0];
        N = [0 1];
        d.n = 4;
        d.m = 1;
        d.r = 2;
    case 'acrobot'
        %%% Acrobot
        l=1;
        gr=9.81;
        m1=1;
        m2=1;
        lc = l/2;
        I1 = 1;
        I2 = 1;
        syms t1 t2 td1 td2 u real
        q = [t1; t2];
        dq = [td1; td2];
        vars = [q; dq];
        D = [I1+I2+m2*l^2+2*m2*l*lc*cos(t2) I2+m2*l*lc*cos(t2); I2+m2*l*lc*cos(t2) I2];
        Dq = @(in)[I1+I2+m2*l^2+2*m2*l*lc*cos(in(2,:)) I2+m2*l*lc*cos(in(2,:)); I2+m2*l*lc*cos(in(2,:)) I2];
        C = [-2*m2*l*lc*sin(t2)*td2 -m2*l*lc*sin(t2)*td2; m2*l*lc*sin(t2)*td1 0];
        G = -[m1*gr*lc*sin(t1)+m2*gr*(l*sin(t1)+lc*sin(t1+t2)); m2*gr*lc*sin(t1+t2)];
        H = C*[td1; td2] + G;
        B = [0;1];
        N = [1 0];
        d.n = 4;
        d.m = 1;
        d.r = 2;
    case 'cartpole_unstable'
        %%% Cartpole, unstable damping
        l = 1;
        gr = 9.81;
        mc = 1;
        mp = 1;
        gam = 1;
        syms x th dx dth u real
        % syms l gr mc mp gam real
        q = [x; th];
        dq = [dx; dth];
        vars = [q; dq];
        D = [mc+mp mp*l*cos(th); mp*l*cos(th) mp*l^2];
        Dq = @(in)[mc+mp mp*l*cos(in(2, :)); mp*l*cos(in(2, :)) mp*l^2];
        % Only difference from nominal cartpole - cubic destabilizing
        % damping
        H = [-mp*l*dth^2*sin(th); -mp*l*gr*sin(th)-gam*dth^3];
        B = [1; 0];
        N = [0 1];
        d.n = 4;
        d.m = 1;
        d.r = 2;
    case 'acrobot_unstable'
        %%% Acrobot
        l=1;
        gr=9.81;
        m1=1;
        m2=1;
        lc = l/2;
        I1 = 1;
        I2 = 1;
        gam = 1;
        syms t1 t2 td1 td2 u real
        q = [t1; t2];
        dq = [td1; td2];
        vars = [q; dq];
        D = [I1+I2+m2*l^2+2*m2*l*lc*cos(t2) I2+m2*l*lc*cos(t2); I2+m2*l*lc*cos(t2) I2];
        Dq = @(in)[I1+I2+m2*l^2+2*m2*l*lc*cos(in(2,:)) I2+m2*l*lc*cos(in(2,:)); I2+m2*l*lc*cos(in(2,:)) I2];
        C = [-2*m2*l*lc*sin(t2)*td2 -m2*l*lc*sin(t2)*td2; m2*l*lc*sin(t2)*td1 0];
        G = -[m1*gr*lc*sin(t1)+m2*gr*(l*sin(t1)+lc*sin(t1+t2)); m2*gr*lc*sin(t1+t2)];
        damp = [-gam * t1^3; 0];
        H = C*[td1; td2] + G + damp;
        B = [0;1];
        N = [1 0];
        d.n = 4;
        d.m = 1;
        d.r = 2;
end

%%% Dynamics
f = [dq; -D \H];
g = [zeros(size(B)); D \ B];

d.f = matlabFunction(f,'vars',{vars});
d.g = matlabFunction(g,'vars',{vars});
d.Df = matlabFunction(jacobian(f,vars),'vars',{vars});
Dg = jacobian(g,vars);
d.Dg = matlabFunction(Dg,'vars',{vars});

%%% Linearizations (For MPC)
df_dx = jacobian(f, vars);
dg_dxu = jacobian(g * u, vars);
d.df_dx = matlabFunction(df_dx,'vars',{vars});
d.dg_dxu = matlabFunction(dg_dxu,'vars',{vars, u});

y = B'*q;
dy = jacobian(y,vars)*f;
Lf2y = jacobian(dy,vars)*f;
LgLfy = jacobian(dy,vars)*g;
z1 = N*q;
z2 = N*D*dq;
d.M = B';

d.y = matlabFunction(y,'vars',{vars});
d.dy = matlabFunction(dy,'vars',{vars});
d.Lf2y = matlabFunction(Lf2y,'vars',{vars});
d.LgLfy = matlabFunction(LgLfy,'vars',{vars});
d.z1 = matlabFunction(z1,'vars',{vars});
d.z2 = matlabFunction(z2,'vars',{vars});
d.n1 = d.y;
d.n2 = d.dy;

Phi = [y; dy; z1; z2];
dPhi = jacobian(Phi, vars);
d.Phi = matlabFunction(Phi,'vars',{vars});
d.dPhi = matlabFunction(dPhi,'vars',{vars});

f_hat_x = [eye(2) zeros(2)]*dPhi*f;
g_hat_x = [eye(2) zeros(2)]*dPhi*g;
omega_x = [zeros(2) eye(2)]*dPhi*f;
assert(all(eval(simplify([zeros(2) eye(2)]*dPhi*g)) == [0; 0])); % zero d had better not have control input.

syms n1_ n2_ z1_ z2_ real
z_vars = [n1_; n2_; z1_; z2_];

Phi_inv = [B * n1_ + N' * z1_; B * n2_ + N' * ((N * Dq(B * n1_ + N' * z1_) * N') \ (z2_ - B' * Dq(B * n1_ + N' * z1_) * N' * n2_))];
d.Phi_inv = matlabFunction(Phi_inv,'vars',{z_vars});

assert(all(all(eval(jacobian(subs(Phi,vars,Phi_inv),z_vars)) == eye(4)))); % phi inverse is correct

omega = simplify(subs(omega_x, vars, Phi_inv));
f_hat = simplify(subs(f_hat_x, vars, Phi_inv));
g_hat = simplify(subs(g_hat_x, vars, Phi_inv));
d.omega = matlabFunction(omega,'vars',{z_vars});
d.f_hat = matlabFunction(f_hat,'vars',{z_vars});
d.g_hat = matlabFunction(g_hat,'vars',{z_vars});

Domega = jacobian(omega, z_vars);
Df_hat = jacobian(f_hat,z_vars);
Dg_hat = jacobian(g_hat,z_vars);
Lf_omega = Domega * [f_hat; omega];
Lg_omega = Domega * [g_hat; 0; 0];

d.Domega = matlabFunction(Domega,'vars',{z_vars});
d.Df_hat = matlabFunction(Df_hat,'vars',{z_vars});
d.Dg_hat = matlabFunction(Dg_hat,'vars',{z_vars});
d.Dg_hatu = matlabFunction(Dg_hat * u,'vars',{z_vars, u});
d.Lf_omega = matlabFunction(Lf_omega ,'vars',{z_vars});
d.Lg_omega = matlabFunction(Lg_omega ,'vars',{z_vars});

end