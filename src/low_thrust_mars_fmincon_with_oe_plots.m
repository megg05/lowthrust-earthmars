%% low_thrust_mars_fmincon.m
%  Direct shooting low-thrust Earth->Mars with fmincon
%  Features: grid search, smooth objective, position+velocity matching
clear; clc; close all;

%% CONSTANTS
mu = 1.32712440018e11; AU = 1.496e8; g0 = 9.80665e-3; d2r = pi/180;

%% SPACECRAFT
m0 = 1500; T_max = 0.6; Isp = 3000;
T_km = T_max*1e-3; c_e = Isp*g0;

%% LOAD EPHEMERIDES
[jd_E, rE, vE] = read_horizons('horizons_results_earth_heliocentric_state_vector.txt');
[jd_M, rM, vM] = read_horizons('horizons_results_mars_heliocentric_state_vector.txt');
fprintf('Ephemeris loaded: Earth %d pts, Mars %d pts\n', length(jd_E), length(jd_M));

%% GRID SEARCH: find best launch date / TOF combination
fprintf('\n=== GRID SEARCH ===\n');

launch_base = 2452791.5;  % 2003-Jun-01
launch_offsets = -90:30:90;  % days offset
tof_range = [400 450 500 550 600];

N_seg = 20;
grid_results = [];

for il = 1:length(launch_offsets)
    for it = 1:length(tof_range)
        lJD = launch_base + launch_offsets(il);
        tof_d = tof_range(it);
        tof_s = tof_d*86400;
        dt_seg = tof_s/N_seg;

        [~,ie] = min(abs(jd_E - lJD));
        [~,im] = min(abs(jd_M - (lJD+tof_d)));
        if ie<1||ie>length(jd_E)||im<1||im>length(jd_M); continue; end

        r0_g = rE(ie,:)'; v0_g = vE(ie,:)';
        rT_g = rM(im,:)'; vT_g = vM(im,:)';

        % Quick Stage 1 with few iterations
        x0_g = build_initial_guess(r0_g, v0_g, rT_g, N_seg);
        opts_g = optimoptions('fmincon','Display','off','Algorithm','sqp',...
            'MaxIterations',200,'MaxFunctionEvaluations',50000);
        [~,fval_g,ef_g] = fmincon(...
            @(x) obj_pos_error(x,N_seg,dt_seg,r0_g,v0_g,m0,T_km,c_e,mu,rT_g,AU),...
            x0_g,[],[],[],[],- ones(3*N_seg,1),ones(3*N_seg,1),...
            @(x) throttle_only(x,N_seg), opts_g);

        grid_results = [grid_results; launch_offsets(il), tof_d, fval_g, ef_g];
        fprintf('  Launch %+4d d | TOF %3d d | err^2 = %.4e | ef=%d\n',...
            launch_offsets(il), tof_d, fval_g, ef_g);
    end
end

% Pick best
[~,best_idx] = min(grid_results(:,3));
best_launch_offset = grid_results(best_idx,1);
best_tof = grid_results(best_idx,2);
fprintf('\nBest: launch offset = %+d days, TOF = %d days (err^2=%.2e)\n',...
    best_launch_offset, best_tof, grid_results(best_idx,3));

%% SETUP WITH BEST PARAMETERS
launch_JD = launch_base + best_launch_offset;
TOF_days = best_tof;
TOF = TOF_days*86400;
dt_seg = TOF/N_seg;

[~, idx_E] = min(abs(jd_E - launch_JD));
r0 = rE(idx_E,:)'; v0 = vE(idx_E,:)';
arrival_JD = launch_JD + TOF_days;
[~, idx_M] = min(abs(jd_M - arrival_JD));
rT = rM(idx_M,:)'; vT = vM(idx_M,:)';

fprintf('\nFinal setup:\n');
fprintf('  Launch: JD %.1f\n', launch_JD);
fprintf('  Arrival: JD %.1f (TOF=%d days)\n', arrival_JD, TOF_days);

%% INITIAL GUESS
n_var = 3*N_seg;
x0 = build_initial_guess(r0, v0, rT, N_seg);
lb = -ones(n_var,1); ub = ones(n_var,1);

%% STAGE 1: Minimize position error
fprintf('\n=== STAGE 1: Find feasible trajectory ===\n');

opts1 = optimoptions('fmincon',...
    'Display','iter','Algorithm','sqp',...
    'MaxFunctionEvaluations',500000,'MaxIterations',5000,...
    'OptimalityTolerance',1e-10,'StepTolerance',1e-14,...
    'ScaleProblem',true);

[x1, fval1, ef1] = fmincon(...
    @(x) obj_pos_error(x,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu,rT,AU),...
    x0,[],[],[],[],lb,ub,...
    @(x) throttle_only(x,N_seg), opts1);

state1 = propagate_full(x1,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu);
err1_pos = norm(state1(1:3)-rT);
err1_vel = norm(state1(4:6)-vT);
fprintf('\nStage 1: pos err=%.0f km (%.6f AU), vel err=%.3f km/s, exit=%d\n',...
    err1_pos, err1_pos/AU, err1_vel, ef1);

%% STAGE 2: Minimize fuel with position+velocity constraints
fprintf('\n=== STAGE 2: Minimize fuel (pos+vel match) ===\n');

opts2 = optimoptions('fmincon',...
    'Display','iter','Algorithm','sqp',...
    'MaxFunctionEvaluations',500000,'MaxIterations',5000,...
    'OptimalityTolerance',1e-8,'ConstraintTolerance',1e-4,...
    'ScaleProblem',true,'StepTolerance',1e-14);

[x_opt, fval2, ef2] = fmincon(...
    @(x) obj_fuel_smooth(x,N_seg),...
    x1,[],[],[],[],lb,ub,...
    @(x) full_constraints(x,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu,rT,vT,AU),...
    opts2);

fprintf('\nStage 2 exit flag: %d\n', ef2);

%% STAGE 3 (optional): refine with tighter tolerance
fprintf('\n=== STAGE 3: Refine ===\n');

opts3 = optimoptions('fmincon',...
    'Display','iter','Algorithm','sqp',...
    'MaxFunctionEvaluations',500000,'MaxIterations',3000,...
    'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-6,...
    'ScaleProblem',true,'StepTolerance',1e-14);

[x_opt, fval3, ef3] = fmincon(...
    @(x) obj_fuel_smooth(x,N_seg),...
    x_opt,[],[],[],[],lb,ub,...
    @(x) full_constraints(x,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu,rT,vT,AU),...
    opts3);

fprintf('\nStage 3 exit flag: %d\n', ef3);

%% POST-PROCESS
[~, traj] = propagate_trajectory(x_opt,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu);
m_final = traj(end,7);
r_final = traj(end,1:3)';
v_final = traj(end,4:6)';
pos_err = norm(r_final-rT);
vel_err = norm(v_final-vT);

fprintf('\n============ RESULTS ============\n');
fprintf('Launch: JD %.1f (offset %+d days)\n', launch_JD, best_launch_offset);
fprintf('TOF: %.0f days\n', TOF_days);
fprintf('Final mass: %.1f kg (of %d kg)\n', m_final, m0);
fprintf('Propellant: %.1f kg\n', m0-m_final);
fprintf('DV: %.3f km/s\n', c_e*log(m0/m_final));
fprintf('Position error: %.1f km (%.6f AU)\n', pos_err, pos_err/AU);
fprintf('Velocity error: %.4f km/s\n', vel_err);
fprintf('V_infinity at Mars: %.3f km/s\n', vel_err);
fprintf('=================================\n');

%% PLOT
figure('Position',[100 100 1600 800]);

% Trajectory
subplot(2,3,1);
plot(traj(:,1)/AU, traj(:,2)/AU, 'b', 'LineWidth',2); hold on;
for k=1:200
    jd_k = launch_JD + TOF_days*k/200;
    [~,ie]=min(abs(jd_E-jd_k)); [~,im]=min(abs(jd_M-jd_k));
    e_orb(k,:)=rE(ie,:)/AU; m_orb(k,:)=rM(im,:)/AU;
end
plot(e_orb(:,1),e_orb(:,2),'g--','LineWidth',1);
plot(m_orb(:,1),m_orb(:,2),'r--','LineWidth',1);
plot(r0(1)/AU,r0(2)/AU,'go','MarkerSize',12,'MarkerFaceColor','g');
plot(rT(1)/AU,rT(2)/AU,'rs','MarkerSize',12,'MarkerFaceColor','r');
plot(0,0,'yo','MarkerSize',14,'MarkerFaceColor','y');
xlabel('x [AU]'); ylabel('y [AU]');
title('Transfer Trajectory'); grid on; axis equal;
legend('SC','Earth','Mars','Location','best');

% Throttle
subplot(2,3,2);
throttle = zeros(N_seg,1);
for i=1:N_seg
    u=[x_opt(i);x_opt(N_seg+i);x_opt(2*N_seg+i)];
    throttle(i)=min(norm(u),1);
end
t_mid=((1:N_seg)-0.5)*dt_seg/86400;
bar(t_mid, throttle*100, 1, 'FaceColor',[0.3 0.6 1]);
xlabel('Days'); ylabel('Throttle [%]'); title('Thrust Profile');
ylim([0 110]); grid on;

% Thrust direction in ecliptic
subplot(2,3,3);
for i=1:N_seg
    quiver_x(i) = x_opt(i);
    quiver_y(i) = x_opt(N_seg+i);
end
quiver(t_mid(:), zeros(N_seg,1), quiver_x(:), quiver_y(:), 0.5);
xlabel('Days'); ylabel(''); title('Thrust Direction (x-y)'); grid on;

% Mass
subplot(2,3,4);
N_sub=20;
t_traj = linspace(0,TOF_days,N_seg*N_sub+1);
plot(t_traj, traj(:,7),'b','LineWidth',1.5);
xlabel('Days'); ylabel('Mass [kg]'); title('Mass History'); grid on;

% Distance from Sun
subplot(2,3,5);
r_mag = sqrt(traj(:,1).^2 + traj(:,2).^2 + traj(:,3).^2)/AU;
plot(t_traj, r_mag,'b','LineWidth',1.5);
xlabel('Days'); ylabel('r [AU]'); title('Heliocentric Distance'); grid on;

% Grid search results
subplot(2,3,6);
if ~isempty(grid_results)
    scatter(grid_results(:,1), grid_results(:,2), 60, log10(grid_results(:,3)+1e-20), 'filled');
    colorbar; xlabel('Launch offset [days]'); ylabel('TOF [days]');
    title('Grid Search (log10 err^2)'); grid on;
    hold on;
    plot(best_launch_offset, best_tof, 'rp', 'MarkerSize', 20, 'MarkerFaceColor','r');
end

sgtitle(sprintf('Low-Thrust Earth-Mars | Launch %+dd | TOF=%dd | m_f=%.0fkg | DV=%.2fkm/s | Vinf=%.2fkm/s',...
    best_launch_offset, TOF_days, m_final, c_e*log(m0/m_final), vel_err));


%% ORBITAL ELEMENTS FROM TRAJECTORY
fprintf('Computing orbital elements along trajectory...\n');
N_pts = size(traj,1);
oe_traj = zeros(N_pts, 6);  % [a, e, i, Om, w, ta]

for k = 1:N_pts
    r_k = traj(k,1:3)';
    v_k = traj(k,4:6)';
    [a_k, e_k, i_k, Om_k, w_k, ta_k] = rv2coe(r_k, v_k, mu);
    oe_traj(k,:) = [a_k, e_k, i_k, Om_k, w_k, ta_k];
end

% Mars elements at arrival
[aM, eM, iM, OmM, wM, ~] = rv2coe(rT, vT, mu);

% Time vector
t_traj = linspace(0, TOF_days, N_pts);

% Plot orbital elements
figure('Position',[100 100 1400 900]);

subplot(2,3,1);
plot(t_traj, oe_traj(:,1)/AU, 'b', 'LineWidth', 1.5); hold on;
yline(aM/AU, 'r--', 'Mars'); yline(oe_traj(1,1)/AU, 'g--', 'Earth');
xlabel('Days'); ylabel('a [AU]'); title('Semi-major Axis'); grid on;

subplot(2,3,2);
plot(t_traj, oe_traj(:,2), 'b', 'LineWidth', 1.5); hold on;
yline(eM, 'r--', 'Mars');
xlabel('Days'); ylabel('e'); title('Eccentricity'); grid on;

subplot(2,3,3);
plot(t_traj, oe_traj(:,3)/d2r, 'b', 'LineWidth', 1.5); hold on;
yline(iM/d2r, 'r--', 'Mars');
xlabel('Days'); ylabel('i [deg]'); title('Inclination'); grid on;

subplot(2,3,4);
plot(t_traj, oe_traj(:,4)/d2r, 'b', 'LineWidth', 1.5); hold on;
yline(OmM/d2r, 'r--', 'Mars');
xlabel('Days'); ylabel('\Omega [deg]'); title('RAAN'); grid on;

subplot(2,3,5);
plot(t_traj, oe_traj(:,5)/d2r, 'b', 'LineWidth', 1.5); hold on;
yline(wM/d2r, 'r--', 'Mars');
xlabel('Days'); ylabel('\omega [deg]'); title('Arg. of Periapsis'); grid on;

subplot(2,3,6);
% Mass on this figure too
plot(t_traj, traj(:,7), 'b', 'LineWidth', 1.5);
xlabel('Days'); ylabel('Mass [kg]'); title('Spacecraft Mass'); grid on;

sgtitle(sprintf('Orbital Elements | Launch +%dd | TOF=%dd | DV=%.2f km/s',...
    best_launch_offset, TOF_days, c_e*log(m0/m_final)));

%% ========== FUNCTIONS ==========

function x0 = build_initial_guess(r0, v0, rT, N_seg)
    v_dir = v0/norm(v0);
    r_dir = (rT-r0)/norm(rT-r0);
    x0 = zeros(3*N_seg,1);
    for i=1:N_seg
        frac = (i-0.5)/N_seg;
        dir_i = (1-frac)*v_dir + frac*r_dir;
        dir_i = dir_i/norm(dir_i);
        x0(i)         = dir_i(1)*0.8;
        x0(N_seg+i)   = dir_i(2)*0.8;
        x0(2*N_seg+i) = dir_i(3)*0.8;
    end
end

function f = obj_pos_error(x, N_seg, dt_seg, r0, v0, m0, T_km, c_e, mu, rT, AU)
    state = propagate_full(x,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu);
    f = sum(((state(1:3)-rT)/AU).^2);
end

function f = obj_fuel_smooth(x, N_seg)
    f = 0;
    for i = 1:N_seg
        u = [x(i); x(N_seg+i); x(2*N_seg+i)];
        f = f + sqrt(u'*u + 1e-10);
    end
end

function [c, ceq] = throttle_only(x, N_seg)
    c = zeros(N_seg,1);
    for i=1:N_seg
        u = [x(i); x(N_seg+i); x(2*N_seg+i)];
        c(i) = u'*u - 1;
    end
    ceq = [];
end

function [c, ceq] = full_constraints(x, N_seg, dt_seg, r0, v0, m0, T_km, c_e, mu, rT, vT, AU)
    state = propagate_full(x,N_seg,dt_seg,r0,v0,m0,T_km,c_e,mu);
    r_final = state(1:3);
    v_final = state(4:6);
    % Position match (scaled by AU) and velocity match (scaled by 30 km/s)
    ceq = [(r_final-rT)/AU; (v_final-vT)/30];
    c = zeros(N_seg,1);
    for i=1:N_seg
        u = [x(i); x(N_seg+i); x(2*N_seg+i)];
        c(i) = u'*u - 1;
    end
end

function state = propagate_full(x, N_seg, dt_seg, r0, v0, m0, T_km, c_e, mu)
    state = [r0; v0; m0];
    N_sub = 20; dt_sub = dt_seg/N_sub;
    for i = 1:N_seg
        u = [x(i); x(N_seg+i); x(2*N_seg+i)];
        for k = 1:N_sub
            f1 = eom(state, u, T_km, c_e, mu);
            f2 = eom(state+0.5*dt_sub*f1, u, T_km, c_e, mu);
            f3 = eom(state+0.5*dt_sub*f2, u, T_km, c_e, mu);
            f4 = eom(state+dt_sub*f3, u, T_km, c_e, mu);
            state = state + (dt_sub/6)*(f1+2*f2+2*f3+f4);
        end
    end
end

function [fuel, traj] = propagate_trajectory(x, N_seg, dt_seg, r0, v0, m0, T_km, c_e, mu)
    N_sub=20; dt_sub=dt_seg/N_sub;
    traj=zeros(N_seg*N_sub+1,7);
    state=[r0;v0;m0]; traj(1,:)=state'; row=1;
    for i=1:N_seg
        u=[x(i);x(N_seg+i);x(2*N_seg+i)];
        for k=1:N_sub
            f1=eom(state,u,T_km,c_e,mu);
            f2=eom(state+0.5*dt_sub*f1,u,T_km,c_e,mu);
            f3=eom(state+0.5*dt_sub*f2,u,T_km,c_e,mu);
            f4=eom(state+dt_sub*f3,u,T_km,c_e,mu);
            state=state+(dt_sub/6)*(f1+2*f2+2*f3+f4);
            row=row+1; traj(row,:)=state';
        end
    end
    fuel=m0-state(7);
end

function dydt = eom(state, u_vec, T_km, c_e, mu)
    r=state(1:3); v=state(4:6); m=state(7);
    R=norm(r); u_mag=norm(u_vec);
    if u_mag>0 && m>0
        accel=(T_km/m)*u_vec;
        dm=-T_km*u_mag/c_e;
    else
        accel=[0;0;0]; dm=0;
    end
    dydt=[v; -mu/R^3*r+accel; dm];
end

function [jd,r,v] = read_horizons(filename)
    fid=fopen(filename,'r');
    if fid==-1; error('Cannot open %s',filename); end
    while true; line=fgetl(fid); if contains(line,'$$SOE'); break; end; end
    jd=[]; r=[]; v=[];
    while true
        line=fgetl(fid);
        if contains(line,'$$EOE')||~ischar(line); break; end
        parts=strsplit(strtrim(line),',');
        if length(parts)>=7
            jd(end+1,1)=str2double(parts{1});
            r(end+1,:)=[str2double(parts{3}),str2double(parts{4}),str2double(parts{5})];
            v(end+1,:)=[str2double(parts{6}),str2double(parts{7}),str2double(parts{8})];
        end
    end
    fclose(fid);
end

function [a,e,inc,Om,w,ta] = rv2coe(r, v, mu)
rv=r(:); vv=v(:); R=norm(rv); V=norm(vv);
hv=cross(rv,vv); h=norm(hv); nv=cross([0;0;1],hv); n=norm(nv);
ev=(1/mu)*((V^2-mu/R)*rv-dot(rv,vv)*vv); e=norm(ev);
energy=V^2/2-mu/R;
if abs(e-1)>1e-10; a=-mu/(2*energy); else; a=inf; end
inc=acos(max(-1,min(1,hv(3)/h)));
if n>1e-10; Om=acos(max(-1,min(1,nv(1)/n)));
    if nv(2)<0; Om=2*pi-Om; end; else; Om=0; end
if n>1e-10&&e>1e-10; w=acos(max(-1,min(1,dot(nv,ev)/(n*e))));
    if ev(3)<0; w=2*pi-w; end; else; w=0; end
if e>1e-10; ta=acos(max(-1,min(1,dot(ev,rv)/(e*R))));
    if dot(rv,vv)<0; ta=2*pi-ta; end; else; ta=0; end
end