import numpy as np

def magic_formula_longitudinal(v_rel:float=None,
                               v_tyre:float=None,
                               load_fz:float=None,
                               slip_ratio:float=None,
                               epsilon:float=1e-12,
                               normalized=True,
                               verbose:bool=False,
                               **kwargs
                               ):
    """
    This function calculates the longitudinal tire forces using Pacejka's magic tyre formula.
    The formulation is using Guiggiani's book.
    The standard values are from Pacejkas tire and vehicle dynamics table 3.1
    """
    #tyre_diameter = 2*(0.205*0.60)+15*0.0254 #2x0*inch2m

    B = 12.27
    C = 1.48
    D = 1100 #peak force
    E = 0.07

    def _slip_ratio(v_rel=v_rel,
                    v_tyre=v_tyre,
                    eps=epsilon
                    ):
        """
        Calculates slip ratio based on 2.71 in Guiggiani's
        """
        sigma_x = -v_rel/v_tyre #-v_rel/(abs(v_tyre)+eps)
        
        return sigma_x

    if slip_ratio is not None and (v_tyre is not None or v_rel is not None):
        raise ValueError("Both slip ratio and wheel speed or vehicle vel was input, please choose only one")
    elif v_tyre is None or v_rel is None:
        sigma = slip_ratio
    elif v_tyre is not None and v_rel is not None:
        sigma = _slip_ratio()

    B_sigma = B*sigma
    grip_force = D * np.sin(C*np.arctan(B_sigma - E*(B_sigma - np.arctan(B_sigma))))
    if normalized:
        return grip_force/load_fz
    else:
        return grip_force


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=3)

    ## model parameters

    # tyre and loading parameters
    Fz      = 700                   # Normal load on tyre           [N]
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    v_tyre  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]

    ## Discretization (the grid of what we vary)
    n_v     = 200                   # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * v_tyre   # List of all relative velocities scaled by tyre rolling speed

    # Longitudinal slip
    sigma_x = -v_rel / v_tyre
    #Fs      = magic_formula_longitudinal(v, V_roll)
    Fs      = magic_formula_longitudinal(slip_ratio=sigma_x, load_fz=Fz)
    Fs_truth= np.genfromtxt("Base_setup/Model_library/y_data.csv", dtype=float)
    print(Fs_truth-Fs)
    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(sigma_x, (Fs_truth-Fs), linewidth=1)
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    """
    contact_length = 0.1
    contact_divisions = 100
    velocity_divisions = 200
    wheel_max_vel = 32
    vehicle_max_vel = 16

    wheel_vel_vec = np.arange(-1, -0.0001, 1/velocity_divisions)*wheel_max_vel
    vehicle_vel_vec = np.arange(-1, -0.0001, 1/velocity_divisions)*vehicle_max_vel
    contact_vec = np.arange(0, 1, 1/contact_divisions)*contact_length
    force_grid = np.empty((len(wheel_vel_vec), len(wheel_vel_vec)))

    for wheel_i, wheel_vel in enumerate(wheel_vel_vec):
        for vehicle_i, vehicle_vel in enumerate(vehicle_vel_vec):
            force_grid[wheel_i, vehicle_i] = magic_formula_longitudinal(wheel_vel, vehicle_vel, 0, False)

    print(force_grid)

    wheel_vel_grid, vehicle_vel_grid = np.meshgrid(wheel_vel_vec, vehicle_vel_vec)

    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(wheel_vel_grid, vehicle_vel_grid, force_grid.T)
    fig.legend()
    plt.grid()
    plt.show()
    """











