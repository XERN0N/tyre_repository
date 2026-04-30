import numpy as np

def magic_formula_longitudinal(vel_tyre:float=None,
                               vel_vehicle:float=None,
                               slip_ratio:float=None,
                               verbose:bool=False,
                               epsilon:float=1e-12,
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
    D = -1100 #peak force
    E = 0.07

    def _slip_ratio(vel_tyre=vel_tyre,
                    vel_vehicle=vel_vehicle,
                    verbose=verbose,
                    eps=epsilon
                    ):
        """
        Calculates slip ratio based on 2.71 in Guiggiani's
        """
        sigma_x = (vel_vehicle-vel_tyre)/(abs(vel_tyre)+eps)

        if verbose:
            if sigma_x>0:
                print(f"The slip ratio is {sigma_x} and is braking")
            elif sigma_x<0:
                print(f"The slip ratio is {sigma_x} and is driving")
            elif sigma_x==0:
                print(f"The slip ratio is {sigma_x} and is rolling")
            else:
                raise RuntimeError(f"Incorrect slip ratio of {sigma_x}")
        
        return sigma_x

    if slip_ratio is not None and (vel_tyre is not None or vel_vehicle is not None):
        raise ValueError("Both slip ratio and wheel speed or vehicle vel was input, please choose only one")
    elif vel_tyre is None or vel_vehicle is None:
        sigma = slip_ratio
    elif vel_tyre is not None and vel_vehicle is not None:
        sigma = _slip_ratio()
    B_sigma = B*sigma
    grip_force = D * np.sin(C*np.arctan(B_sigma - E*(B_sigma - np.arctan(B_sigma))))

    return grip_force   


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=3)

    ## model parameters

    # tyre parameters
    L       = 0.1                   # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                   # Bristle micro-stiffness       [1/m]       range [100-600]
    V_roll  = 16                    # Tyre rolling speed            [m/s]       range [0.1-100]


    ## Discretization (the grid of what we vary)
    n_v     = 200                   # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * V_roll   # List of all relative velocities
    v       = V_roll-v_rel                       # List of tyre velocities

    # Longitudinal slip
    sigma_x = -v_rel / V_roll
    #Fs      = magic_formula_longitudinal(v, V_roll)
    Fs      = magic_formula_longitudinal(slip_ratio=-sigma_x)


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











