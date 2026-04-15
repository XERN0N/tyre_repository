import numpy as np
import matplotlib.pyplot as plt
from MF_model import magic_formula_longitudinal
from Basic_brush_model import basic_brush
from scipy.optimize import least_squares

np.set_printoptions(suppress=True, precision=3)

## model parameters

# tyre parameters
L       = 0.1                                   # contact patch length          [m]         range [0.05-0.2]
k_0     = 240                                   # Bristle micro-stiffness       [1/m]       range [100-600]
V_roll  = 16                                    # Tyre rolling speed            [m/s]       range [0.1-100]
F_z     = 700                                   # Normal load                   [N]         range N/A
epsilon = float(1e-12)                          # regularization parameter      [m^2/s^2]   range N/A

# friction parameters
mu_d    = 0.7                                   # Dynamic friction coefficient  [-]         range (0,1]
mu_s    = 1.2                                   # Static friction coefficient   [-]         range [0.4,2]
v_S     = 3.5                                   # Stribeck velocity             [m/s]       range [2-10]
delta_S = 0.6                                   # Stribeck exponent             [-]         range [0.1,2]

## Discretization (the grid of what we vary)

n_x     = 100                                   # Spatial grid
n_v     = 200                                   # Velocity grid

v_rel   = np.linspace(-1, 0, n_v) * V_roll      # List of all relative velocities
v       = V_roll-v_rel                          # List of tyre velocities
xi      = np.linspace(0, 1, n_x) * L            # List of all spatial positions

# Initialising functions (preparing arrays of zero for functions)
mu      = np.zeros(n_v)                         # Friction coefficients          [-]
z       = np.zeros((n_v,n_x))                   # Bristle deflection, at all velocities and along longditudinal direction

# Longitudinal slip
sigma_x = -v_rel / V_roll
# Fs = basic_brush(v, xi, mu_d, mu_s, v_S, delta_S, L, k_0, V_roll, F_z, epsilon)
#Fs      = magic_formula_longitudinal(v, V_roll)
Fs_MF      = magic_formula_longitudinal(slip_ratio=-sigma_x)
Fs_BB      = basic_brush(v, V_roll, F_z)

# Plot
plt.figure(figsize=(7,5))
plt.plot(sigma_x, Fs_MF/1000, linewidth=1)
plt.plot(sigma_x, Fs_BB/1000, linewidth=1)
plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
plt.ylabel(r'Longitudinal force $F_x$ (kN)')
#plt.xlim(0, 1)
#plt.ylim(0, 1.1 * np.min(Fs/1000))
plt.grid(True)
plt.tight_layout()
plt.show()