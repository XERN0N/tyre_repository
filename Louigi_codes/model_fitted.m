clear; close all; clc;
set(0,'defaultTextInterpreter','latex');

%%
Fz = 700;
V = 16;
n_v = 200;
v = linspace(-1,0,n_v)*V;
sigma = -v/V;
y_data = 1100*sin(1.48*atan(12.27*sigma-0.07*(12.27*sigma-atan(12.27*sigma))))/Fz;

figure
plot(sigma,y_data)

%% Model parameters

nParams = 6; % number of parameters
lb = [0.05, 100, 0.7, 1, 0.1, 0.1];  
ub = [0.12, 800, 1.2, 2, 20, 2]; 

%% Run GA
options = optimoptions('ga', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 300, ...
    'Display', 'iter', ...
    'UseParallel', true);

theta_opt = ga(@(theta)objectiveFunction(theta, y_data), ...
               nParams, [], [], [], [], lb, ub, [], options);

disp('Optimized parameters:');
disp(theta_opt);

function cost = objectiveFunction(theta, y_data)
% theta = [sigma, c_c, mu]

% Tire parameters
L = theta(1);         % Contact patch length     (m)                            % Range [0.05-0.2]
k_0 = theta(2);       % Bristle micro-stiffness  (1/m)                          % Range [100-600]
V = 16;          % Tire rolling speed       (m/s)                          % Range [0.1-100]
Fz = 700;       % Normal load              (N)
p = Fz/L;        % Normal pressure distribution (N/m^2)
epsilon = 1e-12; % Regularisation parameter (m^2/s^2)

% Friction parameters
mu_d = theta(3);    % Dynamic friction coefficient (-)                          % Range (0,1]
mu_s = theta(4)*theta(3);    % Static friction coefficient  (-)                          % Range [0.4,2]
v_S = theta(5);     % Stribeck velocity            (m/s)                        % Range [2-10]
delta_S = theta(6); % Stribeck exponent            (-)                          % Range [0.1-2]


%% Discretisation

n_x = 100; % Spatial grid 
n_v = length(y_data); % Velocity grid

v = linspace(-1,0,n_v)*V;
xi = linspace(0,1,n_x)*L;
delta_xi = xi(2)-xi(1);

% Initialise functions 
mu = zeros(n_v);    % Friction coefficient (-)
z = zeros(n_v,n_x); % Bristle deflection   (m)

for i=1:n_v
    for j=1:n_x
        mu(i) = mu_d + (mu_s-mu_d)*exp(-abs(v(i)/v_S)^delta_S);
        z(j,i) = -mu(i)/k_0*v(i)/sqrt(v(i)^2+epsilon)*(1 ...
            -exp(-sqrt(v(i)^2+epsilon)/V*k_0/mu(i)*xi(j)));
    end
end

% Compute tyre force
F = sum(z)*delta_xi*k_0*p;
y = F/Fz;


cost = sum((y - y_data).^2);


end

%%

% tetha_opt = [0.1200  600.0000    1.2000    2.0000    4.4429    0.8154];

% Tire parameters
L = theta_opt(1);         % Contact patch length     (m)                            % Range [0.05-0.2]
k_0 = theta_opt(2);       % Bristle micro-stiffness  (1/m)                          % Range [100-600]
V = 16;          % Tire rolling speed       (m/s)                          % Range [0.1-100]
Fz = 700;       % Normal load              (N)
p = Fz/L;        % Normal pressure distribution (N/m^2)
epsilon = 1e-12; % Regularisation parameter (m^2/s^2)

% Friction parameters
mu_d = theta_opt(3);    % Dynamic friction coefficient (-)                          % Range (0,1]
mu_s = theta_opt(4)*theta_opt(3);    % Static friction coefficient  (-)                          % Range [0.4,2]
v_S = theta_opt(5);     % Stribeck velocity            (m/s)                        % Range [2-10]
delta_S = theta_opt(6); % Stribeck exponent            (-)                          % Range [0.1-2]


% Discretisation

n_x = 100; % Spatial grid 
n_v = length(y_data); % Velocity grid

v = linspace(-1,0,n_v)*V;
xi = linspace(0,1,n_x)*L;
delta_xi = xi(2)-xi(1);

% Initialise functions 
mu = zeros(n_v);    % Friction coefficient (-)
z = zeros(n_v,n_x); % Bristle deflection   (m)

for i=1:n_v
    for j=1:n_x
        mu(i) = mu_d + (mu_s-mu_d)*exp(-abs(v(i)/v_S)^delta_S);
        z(j,i) = -mu(i)/k_0*v(i)/sqrt(v(i)^2+epsilon)*(1 ...
            -exp(-sqrt(v(i)^2+epsilon)/V*k_0/mu(i)*xi(j)));
    end
end

% Compute tyre force
F = sum(z)*delta_xi*k_0*p;
y = F/Fz;


%% Figures



% Tyre force vs slip
figure(1)
plot(sigma,F, 'LineWidth', 1)
hold on
hold on
plot(sigma,y_data*Fz, 'k--', 'LineWidth', 1)
% hold on
% plot(sigma,y*4000, 'k--', 'LineWidth', 1)
% hold on
% plot(sigma,y*6000, 'k--', 'LineWidth', 1)
% set(gca,'TickLabelInterpreter','latex')
% xlabel('Longitudinal slip $\sigma_x$ (-)')
% ylabel('Longitudinal force $F_x$ (kN)')
% % axis([0 1 0 1.1*max(F/1000)])



% % Tyre force vs slip
% figure(1)
% plot(sigma,y_data, 'LineWidth', 1)
% hold on
% plot(sigma,y, 'LineWidth', 1)
% % hold on
% % plot(sigma,y*2000, 'LineWidth', 1)
% % hold on
% % plot(sigma,y*3000, 'LineWidth', 1)
% % hold on
% % plot(sigma,y*4000, 'LineWidth', 1)
% set(gca,'TickLabelInterpreter','latex')
% xlabel('Longitudinal slip $\sigma_x$ (-)')
% ylabel('Longitudinal force $F_x$ (kN)')
% % axis([0 1 0 1.1*max(F/1000)])
