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
    """Compute longitudinal tyre force using Pacejka's Magic Formula (Guiggiani formulation).

    Implements the standard sine-arctan Magic Formula:

        F = D · sin(C · arctan(B·σ - E·(B·σ - arctan(B·σ))))

    where σ is the longitudinal slip ratio.  Tyre coefficients B, C, D, E are
    hardcoded from Pacejka's *Tyre and Vehicle Dynamics*, Table 3.1.

    Slip ratio is computed as ``σ = -v_rel / v_tyre`` (Guiggiani eq. 2.71) when
    ``v_rel`` and ``v_tyre`` are supplied.  Alternatively, pass ``slip_ratio``
    directly.  Supplying both raises ``ValueError``.

    Args:
        v_rel:      Relative sliding velocity between tyre and road (``v_wheel - v_vehicle``),
                    scalar or array [m/s].  Negative during braking.
        v_tyre:     Tyre rolling (peripheral) velocity [m/s].  Required when
                    ``v_rel`` is provided.
        load_fz:    Normal load on the tyre [N].  Required when ``normalized=True``.
        slip_ratio: Pre-computed longitudinal slip ratio σ [-].  Use instead of
                    ``v_rel`` + ``v_tyre`` when slip is already known.
        epsilon:    Small regularisation constant to guard against division by
                    zero in the slip calculation [m/s].  Default ``1e-12``.
        normalized: If ``True`` (default), divide the force by ``load_fz`` and
                    return a dimensionless coefficient.  If ``False``, return the
                    raw force in [N].
        verbose:    Reserved for future diagnostic output.  Currently unused.
        **kwargs:   Accepted but ignored, for forward-compatibility.

    Returns:
        Longitudinal force or normalised force coefficient, matching the shape
        of the input slip array.  Scalar in, scalar out; array in, array out.

    Raises:
        ValueError: If both ``slip_ratio`` and (``v_rel`` or ``v_tyre``) are given.

    Notes:
        Hardcoded Pacejka coefficients (Table 3.1):
            B = 12.27, C = 1.48, D = 1100 N, E = 0.07
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
    import scienceplots
    plt.style.use(["science","no-latex", "grid", "high-vis"])
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
    sigma_y = -v_rel / v_tyre
    #Fs      = magic_formula_longitudinal(v, V_roll)
    Fs      = magic_formula_longitudinal(slip_ratio=sigma_y, load_fz=Fz, normalized=True)
    Fs_truth= np.genfromtxt("Base_setup/Model_library/y_data.csv", dtype=float)
    #print(Fs_truth-Fs) #Debugging print to show residual

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(sigma_y, (Fs), label="Pacejka's Magic Formula", linewidth=1)
    # plt.plot(sigma_x, (Fs_truth-Fs), label="MF - residual", linewidth=1) #residual plot for comparison to luigi
    plt.xlabel(r'Lateral slip $\sigma_y$ [deg]')
    plt.ylabel(r'Lateral force normalized $F_y$ [-]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("MF_baseline.png")
    plt.show()