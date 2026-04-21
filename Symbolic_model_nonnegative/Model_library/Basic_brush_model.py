import numpy as np

def basic_brush(vel_tyre:float,
                vel_vehicle:float,
                load_fz:float=1000, #700
                mu_d:float=0.7,
                mu_s:float=1.2,
                vel_stribeck:float=3.5,
                exp_stribeck:float=0.6,
                contact_len:float=0.1,
                k_bristle:float=240.0,
                num_bristle:int=100,
                epsilon=float(1e-12),
                return_z:bool=False,
                ):
    """
    This function uses stribeck friction applied to a simple brush model to obtain longitudinal forces.
    """
    
    p= load_fz/contact_len                              # normal pressure ditribution                       [N/m^2]     range N/A
    bristle_spacing= contact_len/num_bristle            # the spatial difference of positions in xi         range N/A
    bristle_pos = np.linspace(0, contact_len, num_bristle)
    vel_relative = vel_vehicle - vel_tyre
    z = np.empty((len(vel_relative),len(bristle_pos)))

    for i, vel_r in enumerate(vel_relative):
        # Calculating the friction for all velocities
        mu = mu_d+(mu_s-mu_d)*np.exp(-np.abs(vel_r/vel_stribeck)**exp_stribeck)
        for j, bristle in enumerate(bristle_pos):
            # Calculating the deflection
            z[i,j] = -vel_r/abs(vel_r+epsilon)*mu/k_bristle \
                *(1-np.exp(-np.sqrt(vel_r**2+epsilon)*k_bristle/(vel_vehicle*mu)*bristle))
    z_sum = np.sum(z,1)

    if return_z:
        return z_sum*bristle_spacing*p*k_bristle, z_sum
    else:
        return z_sum*bristle_spacing*p*k_bristle 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ## model parameters

    # tyre parameters
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    V_roll  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]
    F_z     = 3000                  # Normal load                   [N]         range N/A
    epsilon = float(1e-12)          # regularization parameter      [m^2/s^2]   range N/A

    # friction parameters
    mu_d    = 0.7                   # Dynamic friction coefficient  [-]         range (0,1]
    mu_s    = 1.2                   # Static friction coefficient   [-]         range [0.4,2]
    v_S     = 3.5                   # Stribeck velocity             [m/s]       range [2-10]
    delta_S = 0.6                   # Stribeck exponent             [-]         range [0.1,2]

    ## Discretization (the grid of what we vary)

    n_x     = 100                   # Spatial grid
    n_v     = 200                   # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * V_roll   # List of all relative velocities
    v       = V_roll-v_rel                       # List of tyre velocities
    xi      = np.linspace(0, 1, n_x) * L         # List of all spatial positions

    # Initialising functions (preparing arrays of zero for functions)
    mu      = np.zeros(n_v)         # Friction coefficients          [-]
    z       = np.zeros((n_v,n_x))     # Bristle deflection, at all velocities and along longditudinal direction

    # Fs = basic_brush(v, xi, mu_d, mu_s, v_S, delta_S, L, k_0, V_roll, F_z, epsilon)
    Fs      = basic_brush(v, V_roll, F_z)

    # Longitudinal slip
    sigma_x = -v_rel / V_roll

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(sigma_x, Fs/1000, linewidth=1)
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
