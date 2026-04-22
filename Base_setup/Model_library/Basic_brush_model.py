import numpy as np

def basic_brush(v_rel:float,
                v_tyre:float,
                load_fz:float=700,
                mu_d:float=0.7,
                mu_s:float=1.2,
                vel_stribeck:float=3.5,
                exp_stribeck:float=0.6,
                contact_len:float=0.1,
                k_bristle:float=240.0,
                num_bristle:int=100,
                epsilon=float(1e-12),
                normalized=True,
                return_z:bool=False,
                ):
    """
    This function uses stribeck friction applied to a simple brush model to obtain longitudinal forces.
    """
    
    p = load_fz/contact_len                              # normal pressure ditribution                       [N/m^2]     range N/A
    bristle_pos = np.linspace(0.0, 1.0, num_bristle)*contact_len
    bristle_spacing = bristle_pos[1]-bristle_pos[0]#contact_len/num_bristle            # the spatial difference of positions in xi         range N/A
    z = np.empty((len(v_rel),len(bristle_pos)))

    for i, v_rel_scalar in enumerate(v_rel):
        # Calculating the friction for all velocities
        mu = mu_d+(mu_s-mu_d)*np.exp(-np.abs(v_rel_scalar/vel_stribeck)**exp_stribeck)
        for j, bristle in enumerate(bristle_pos):
            # Calculating the deflection
            z[i,j] = -v_rel_scalar/(abs(v_rel_scalar)+epsilon)*mu/k_bristle \
                *(1-np.exp(-np.sqrt(v_rel_scalar**2+epsilon)*k_bristle/(v_tyre*mu)*bristle))
    z_sum = np.sum(z,1)

    F=z_sum*bristle_spacing*p*k_bristle
    if normalized:
        F=F/load_fz
    if return_z:
        return F, z_sum
    else:
        return F 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=3)
    ## model parameters

    # tyre parameters
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    v_tyre  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]
    Fz     = 700                  # Normal load                   [N]         range N/A
    epsilon = float(1e-12)          # regularization parameter      [m^2/s^2]   range N/A

    # friction parameters
    mu_d    = 0.7                   # Dynamic friction coefficient  [-]         range (0,1]
    mu_s    = 1.2                   # Static friction coefficient   [-]         range [0.4,2]
    v_S     = 3.5                   # Stribeck velocity             [m/s]       range [2-10]
    delta_S = 0.6                   # Stribeck exponent             [-]         range [0.1,2]

    ## Discretization (the grid of what we vary)

    n_x     = 100                   # Spatial grid
    n_v     = 200                   # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * v_tyre   # List of all relative velocities
    xi      = np.linspace(0, 1, n_x) * L         # List of all spatial positions

    ## Evaluation
    Fs      = basic_brush(v_rel=v_rel, v_tyre=v_tyre, load_fz=Fz)
    Fs_truth= np.genfromtxt("Base_setup/Model_library/y_data_brush.csv", dtype=float)
    
    # Longitudinal slip
    sigma_x = -v_rel / v_tyre

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(sigma_x, (Fs), linewidth=1)
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
