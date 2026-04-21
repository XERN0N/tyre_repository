clear; close all; clc;
set(0,'defaultTextInterpreter','latex');


%% DATA

% Data type (0 = longitudinal, 1 = lateral)
dataType = 1;

if dataType == 0
    file = load('C:\Users\how_g\Uni\2_semester\RnD_project\tyre_project_codes\tyre_data_ML\SAE_TTC_tire_data\longitudinal_tire_test.mat');
else
    file = load('C:\Users\how_g\Uni\2_semester\RnD_project\tyre_project_codes\tyre_data_ML\SAE_TTC_tire_data\lateral_tire_test.mat');
end

% Define slip inputs
slip_ratio = file.SR;                             % Practical longitudinal slip   (-)
sigma_x    = slip_ratio./(1+slip_ratio)+0.05;          % Theoretical longitudinal slip (-)
sigma_y    = tan(file.SA*pi/180./(1+slip_ratio)); % Theoretical lateral slip      (-)

% Define forces
Fx = file.FX;       % Longitudinal force (N)
Fy = -file.FY;       % Lateral force (N)
Fz = -file.FZ*4.448; % Vertical force (N)

figure
plot(sigma_x,Fx)


%% BIN DATA BASED ON THE VERTICAL FORCE

% Binning
nBins = 5;
Fz_edges     = linspace(min(Fz), max(Fz), nBins+1);
[~,~,Fz_bin] = histcounts(Fz, Fz_edges);
group_id = Fz_bin;

% Grouped outputs
if dataType == 0
    F_cells     = arrayfun(@(i) Fx(Fz_bin==i), 1:nBins, 'UniformOutput', false);
    sigma_cells = arrayfun(@(i) sigma_x(Fz_bin==i), 1:nBins, 'UniformOutput', false);
else
    F_cells     = arrayfun(@(i) Fy(Fz_bin==i), 1:nBins, 'UniformOutput', false);
    sigma_cells = arrayfun(@(i) sigma_y(Fz_bin==i), 1:nBins, 'UniformOutput', false);
end

% Clean and sort bins
sigma_cells_sorted = cell(size(sigma_cells));
F_cells_sorted    = cell(size(F_cells));

% for i = 1:nBins
%     if isempty(sigma_cells{i}), continue; end
% 
%     % Sort
%     [sigma_i, ord] = sort(sigma_cells{i});
%     F_i = F_cells{i}(ord);
% 
%     % Remove duplicates
%     [sigma_i, ~, ic] = unique(sigma_i);
%     F_i = accumarray(ic, F_i, [], @mean);
% 
%     % Store
%     sigma_cells_sorted{i} = sigma_i;
%     F_cells_sorted{i}     = F_i;
% end

for i = 1:nBins
    if isempty(sigma_cells{i}), continue; end

    % Sort
    [s, ord] = sort(sigma_cells{i});
    f = F_cells{i}(ord);

    % Outliers removal
    % Remove NaNs / infs
    idx_valid = isfinite(s) & isfinite(f);
    s = s(idx_valid);
    f = f(idx_valid);

    % Remove very small slip (noisy region)
    idx_slip = abs(s) > 1e-4;
    s = s(idx_slip);
    f = f(idx_slip);

    % Median Absolute Deviation (MAD)
    med_f = median(f);
    mad_f = median(abs(f - med_f)) + 1e-12; % avoid zero

    % Threshold
    threshold = 2;
    idx_mad = abs(f - med_f) < threshold * mad_f;

    % Apply filter
    s = s(idx_mad);
    f = f(idx_mad);

    % Remove duplicates
    [s, ~, ic] = unique(s);
    f = accumarray(ic, f, [], @mean);

    % Store
    sigma_cells_sorted{i} = s;
    F_cells_sorted{i}     = f;
end

% Vertical force corresponding to each bin
Fz_rep = arrayfun(@(i) mean(Fz(Fz_bin==i)), 1:nBins);


y_data = cell(1, nBins);
x_data = cell(1, nBins);

for i =1:nBins
    y_data{i} = F_cells_sorted{i}/Fz_rep(i);
    x_data{i} = sigma_cells_sorted{i};
end

% Reduce sampling size
n_points = 200;

y_new = cell(1, nBins);
x_new = cell(1, nBins);

for i=1:nBins
    x_new{i} = linspace(min(x_data{i}), max(x_data{i}), n_points)';
    y_new{i} = interp1(x_data{i}, y_data{i}, x_new{i}, 'pchip')';  % shape-preserving
end


%% MODEL OPTIMIZATION

% The following parameters are optimized:
% Load-dependent contact patch length L = theta(1) + theta(2)*sqrt(Fz) (m)
% Load-dependent normalized bristle stiffness k_0 = theta(3)-theta(4)*Fz (1/m)
% Load-dependent dynamic friction coefficient mu_d = theta(5)-theta(6)*Fz
% (-)
% Static friction coefficient mu_s = theta(7)*mu_d (-)
% Stribeck velocity v_S = theta(8) (m/s)
% Stribeck exponent delta_S = theta(9) (-)
% Stribeck parameter a_S = theta(10)*mu_d
% The friction coefficient is modeled as mu = mu_d +
% (mu_s-mu_d)*exp(-|v|^delta_S/v_S) -a_S*tanh(v/v_S)

nParams = 10; % number of parameters
lb = [0.05, 0, 200, 0, 0.5, 1/40000, 1, 2, 0.4, 0];
ub = [0.12, 1/500, 600, 0.04, 0.7, 1/10000, 2.5, 8, 2.5, 0.5];
%  theta_opt = [0.0668, 0.0001, 360.4850, 0.0230, 0.6456, 3.07e-05, 1.7213, 3.6888, 1.3652, 0.0932]

% Run GA
options = optimoptions('ga', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 300, ...
    'Display', 'iter', ...
    'UseParallel', true);

theta_opt = ga(@(theta)objectiveFunction(theta, y_new, x_new, Fz_rep), ...
    nParams, [], [], [], [], lb, ub, [], options);

disp('Optimized parameters:');
disp(theta_opt);

function cost = objectiveFunction(theta, y_data, x_data, Fz_rep)

nBins = length(y_data);
cost = 0;

for k=1:nBins

    x_k = x_data{k};
    y_k = y_data{k};
    Fz = Fz_rep(k);

    % Tire parameters
    L = theta(1)+theta(2)*sqrt(Fz); % Contact patch length     (m)
    k_0 = theta(3)-theta(4)*Fz;     % Bristle micro-stiffness  (1/m)
    V = 16;                         % Tire rolling speed       (m/s)
    p = Fz/L;                       % Normal pressure distribution (N/m^2)
    epsilon = 1e-12;                % Regularisation parameter (m^2/s^2)

    % Friction parameters
    mu_d = theta(5) - theta(6)*Fz; % Dynamic friction coefficient (-)
    mu_s = theta(7)*mu_d;          % Static friction coefficient  (-)
    v_S = theta(8);                % Stribeck velocity            (m/s)
    delta_S = theta(9);            % Stribeck exponent            (-)
    a_S = theta(10)*mu_d;          % Stribeck parameter           (-)

    % Discretization
    n_x = 100; % Spatial grid
    n_v = length(y_k); % Velocity grid

    v = -x_k*V;
    xi = linspace(0,1,n_x)*L;
    delta_xi = xi(2)-xi(1);

    % Initialize functions
    mu = zeros(n_v);    % Friction coefficient (-)
    z = zeros(n_v,n_x); % Bristle deflection   (m)

    for i=1:n_v
        for j=1:n_x
            mu(i) = mu_d + (mu_s-mu_d)*exp(-abs(v(i)/v_S)^delta_S) ...
                - a_S*tanh(v(i)/v_S);


            z(j,i) = -mu(i)/k_0*v(i)/sqrt(v(i)^2+epsilon)*(1 ...
                -exp(-sqrt(v(i)^2+epsilon)/V*k_0/mu(i)*xi(j)));
        end
    end

    % Compute tyre force
    F = sum(z)*delta_xi*k_0*p;
    y = F/Fz;
    cost = cost + sum((y - y_k).^2);

end

end


%% POSTPROCESSING

% Tire parameters
L = theta_opt(1)+theta_opt(2)*sqrt(Fz_rep); % Contact patch length         (m)
k_0 = theta_opt(3)-theta_opt(4)*Fz_rep;     % Bristle micro-stiffness      (1/m)
V = 16;                                     % Tire rolling speed           (m/s)
p = Fz_rep./L;                              % Normal pressure distribution (N/m^2)
epsilon = 1e-12;                            % Regularization parameter (m^2/s^2)

% Friction parameters
mu_d = theta_opt(5)-theta_opt(6)*Fz_rep; % Dynamic friction coefficient (-)
mu_s = theta_opt(7)*mu_d;                % Static friction coefficient  (-)
v_S = theta_opt(8);                      % Stribeck velocity            (m/s)
delta_S = theta_opt(9);                  % Stribeck exponent            (-)
a_S = theta_opt(10)*mu_d;                % Stribeck parameter

% Discretisation

n_x = 100; % Spatial grid
n_v = 200; % Velocity grid

v = linspace(-1,1,n_v)*V;
xi_bar = linspace(0,1,n_x);

% Initialise functions
mu = zeros(nBins, n_v);     % Friction coefficient (-)
z = zeros(nBins, n_v, n_x); % Bristle deflection   (m)
F_b = zeros(nBins, n_v);    % Tire force           (N)

for k =1:nBins

xi = xi_bar*L(k);
delta_xi = xi(2)-xi(1);

    for i=1:n_v
        for j=1:n_x
            mu(k,i) = mu_d(k) + (mu_s(k) ...
                -mu_d(k))*exp(-abs(v(i)/v_S)^delta_S) ...
                -a_S(k)*tanh(v(i)/v_S);


            z(k,j,i) = -mu(k,i)/k_0(k)*v(i)/sqrt(v(i)^2+epsilon)*(1 ...
                -exp(-sqrt(v(i)^2+epsilon)/V*k_0(k)/mu(k,i)*xi(j)));
        end
    end
    F_b(k,:) = sum(z(k,:,:))*delta_xi*k_0(k)*p(k);
end


%% FIGURES

figure(1)
% plot(sigma_cells_sorted{1}, F_cells_sorted{1}/1000, '.', ...
%     'Color', [0.55,0.80,0.95], 'LineStyle', 'none', ...
%     'HandleVisibility','off')
% hold on
% plot(sigma_cells_sorted{2}, F_cells_sorted{2}/1000, '.', ...
%     'Color', [1.00,0.75,0.65], 'LineStyle', 'none', ... 
%     'HandleVisibility','off')
% hold on
% plot(sigma_cells_sorted{3}, F_cells_sorted{3}/1000, '.', ...
%     'Color', [0.95,0.85,0.65], 'LineStyle', 'none', ...
%     'HandleVisibility','off')
% hold on
plot(x_new{1}, y_new{1}*Fz_rep(1)/1000, 'o', ...
    'Color', [0.55,0.80,0.95], 'LineStyle', 'none', ...
    'MarkerFaceColor', [0.55,0.80,0.95], 'MarkerSize', 4, ...
    'HandleVisibility','off')
hold on
plot(x_new{2}, y_new{2}*Fz_rep(2)/1000, 'o', ...
    'Color', [1.00,0.75,0.65], 'LineStyle', 'none', ... 
    'MarkerFaceColor', [1.00,0.75,0.65], 'MarkerSize', 4, ...
    'HandleVisibility','off')
hold on
plot(x_new{3}, y_new{3}*Fz_rep(3)/1000, 'o', ...
    'Color', [0.95,0.85,0.65], 'LineStyle', 'none', ...
    'MarkerFaceColor', [0.95,0.85,0.65], 'MarkerSize', 4, ...
    'HandleVisibility','off')
hold on
plot(x_new{4}, y_new{4}*Fz_rep(4)/1000, 'o', ...
    'Color', [0.75,0.65,0.75], 'LineStyle', 'none', ...
    'MarkerFaceColor', [0.75,0.65,0.75], 'MarkerSize', 4, ...
    'HandleVisibility','off')
hold on
plot(x_new{5}, y_new{5}*Fz_rep(5)/1000, 'o', ...
    'Color', [0.80,0.90,0.75], 'LineStyle', 'none', ...
    'MarkerFaceColor', [0.80,0.90,0.75], 'MarkerSize', 4, ...
    'HandleVisibility','off')
hold on
plot(-v/V,F_b(1,:)/1000, 'Color', [0.00,0.45,0.74], 'LineWidth', 1, ...
    'DisplayName', '$F_z = 1$ kN');
hold on
plot(-v/V,F_b(2,:)/1000, 'Color', [0.85,0.33,0.10], 'LineWidth', 1, ...
    'DisplayName', '$F_z = 2$ kN');
hold on
plot(-v/V,F_b(3,:)/1000, 'Color', [0.93,0.69,0.13], 'LineWidth', 1, ...
    'DisplayName', '$F_z = 3$ kN');
hold on
plot(-v/V,F_b(4,:)/1000, 'Color', [0.49,0.18,0.56], 'LineWidth', 1, ...
    'DisplayName', '$F_z = 4$ kN');
hold on
plot(-v/V,F_b(5,:)/1000, 'Color', [0.47,0.67,0.19], 'LineWidth', 1, ...
    'DisplayName', '$F_z = 5$ kN');
axis([-0.3 0.3 -3.1 3.1])
grid on
set(gca,'TickLabelInterpreter','latex')
xlabel('Lateral slip $\sigma_y$ (-)')
ylabel('Lateral force $F_y$ (kN)')
legend('show', 'Interpreter','latex', 'Location', 'southeast');
xticks([-0.3 -0.2 -0.1 0 0.1 0.2 0.3])
yticks([-3 -2 -1 0 1 2 3])
xticklabels({'$-0.3$','$-0.2$','$-0.1$','0','0.1', '0.2', '0.3'})
yticklabels({'$-3$','$-2$','$-1$','0','1', '2', '3'})