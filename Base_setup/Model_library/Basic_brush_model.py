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
    """Compute lateral tyre force using a basic brush model with Stribeck friction.

    Models the tyre contact patch as a row of elastic bristles anchored to the
    belt.  Each bristle deflects as it travels through the contact zone; once the
    elastic restoring force exceeds the local friction limit it slides.  The
    friction coefficient itself varies with sliding speed via the Stribeck curve:

        μ(v) = μ_d + (μ_s - μ_d) · exp(-|v_rel / vel_stribeck|^exp_stribeck)

    Bristle deflection at position ξ in the contact patch for a given ``v_rel``:

        z(ξ) = -sign(v_rel) · (μ / k_bristle) · (1 - exp(-|v_rel| · k_bristle / (v_tyre · μ) · ξ))

    Normal pressure is assumed uniform: p = Fz / contact_len [N/m].

    Total force is the integral of bristle spring forces over the contact length,
    approximated by a Riemann sum:

        F = Σ_j  z(ξ_j) · bristle_spacing · p · k_bristle

    Args:
        v_rel:         Relative sliding velocity (``v_wheel - v_vehicle``) [m/s],
                       1-D array.  Negative during braking.
        v_tyre:        Tyre rolling (peripheral) velocity [m/s], scalar.
        load_fz:       Normal load on the tyre [N].  Default 700 N.
        mu_d:          Dynamic (Coulomb) friction coefficient [-].  Default 0.7.
        mu_s:          Static friction coefficient at zero sliding speed [-].
                       Must satisfy ``mu_s >= mu_d``.  Default 1.2.
        vel_stribeck:  Stribeck characteristic velocity [m/s]: speed at which
                       friction has decayed roughly 63 % toward ``mu_d``.
                       Default 3.5 m/s.
        exp_stribeck:  Shape exponent of the Stribeck exponential [-].
                       Higher values give a sharper transition.  Default 0.6.
        contact_len:   Contact patch length [m].  Default 0.1 m.
        k_bristle:     Bristle lateral stiffness per unit length [1/m],
                       often denoted k₀.  Default 240 1/m.
        num_bristle:   Number of spatial discretisation points across the contact
                       patch.  Higher values give more accurate force integration
                       at increased cost.  Default 100.
        epsilon:       Regularisation constant [m²/s²] added under the square root
                       to avoid ``sign(0)`` singularity at zero sliding speed.
                       Default 1e-12.
        normalized:    If ``True`` (default), divide the force by ``load_fz`` and
                       return a dimensionless coefficient.  If ``False``, return
                       the raw force in [N].
        return_z:      If ``True``, also return the summed bristle deflection
                       profile (shape ``(n_v,)``).  Default ``False``.

    Returns:
        F:     Force or normalised force coefficient array of shape ``(n_v,)``,
               where ``n_v = len(v_rel)``.
        z_sum: Summed bristle deflection array of shape ``(n_v,)``.  Only returned
               when ``return_z=True``.

    Notes:
        The double loop (velocity x bristle position) is the computational
        bottleneck.  Increase ``num_bristle`` only as far as accuracy demands,
        and prefer vectorisation or JIT compilation for later use.
    """
    
    p = load_fz/contact_len                                                     # normal pressure distribution   [N/m]               range N/A
    bristle_pos = np.linspace(0.0, 1.0, num_bristle)*contact_len
    bristle_spacing = bristle_pos[1]-bristle_pos[0]#contact_len/num_bristle     # the spatial difference of positions in xi         range N/A
    z = np.empty((len(v_rel),len(bristle_pos)))

    for i, v_rel_scalar in enumerate(v_rel):
        # Calculating the friction for all velocities
        mu = mu_d+(mu_s-mu_d)*np.exp(-np.abs(v_rel_scalar/vel_stribeck)**exp_stribeck)
        for j, bristle in enumerate(bristle_pos):
            # Calculating the deflection
            z[i,j] = -v_rel_scalar/(np.sqrt(v_rel_scalar**2+epsilon))*mu/k_bristle \
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
    import scienceplots
    plt.style.use(["science","no-latex", "grid", "high-vis"])
    np.set_printoptions(suppress=True, precision=3)
    ## model parameters

    # tyre parameters
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    v_tyre  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]
    Fz      = 700                   # Normal load                   [N]         range N/A
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
    #xi      = np.linspace(0, 1, n_x) * L         # List of all spatial positions

    ## Evaluation
    Fs      = basic_brush(v_rel=v_rel, v_tyre=v_tyre, num_bristle=n_x, load_fz=Fz)
    Fs_truth= np.genfromtxt("Base_setup/Model_library/y_data_brush.csv", dtype=float)
    
    # Longitudinal slip
    sigma_x = -v_rel / v_tyre

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(sigma_x, (Fs), label="Basic Brush model", linewidth=1)
    plt.plot(sigma_x, (Fs_truth), label="Basic Brush model - Luigi", linewidth=1)
    # plt.plot(sigma_x, (Fs_truth-Fs), label="Basic Brush model residual", linewidth=1)
    plt.xlabel(r'Lateral slip $\sigma_y$ [deg]')
    plt.ylabel(r'Lateral force normalized $F_y$ [-]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Basic_Brush_baseline.png")
    plt.show()
