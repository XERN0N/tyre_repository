import numpy as np
import matplotlib.pyplot as plt
from MF_model import magic_formula_longitudinal
from Basic_brush_model import basic_brush
from scipy.optimize import least_squares
from pysr import PySRRegressor

np.set_printoptions(suppress=True, precision=3)

def residual(params, *args):    
    res_BB = basic_brush(vel_tyre=args[0],
                         vel_vehicle=args[1],
                         mu_d=params[2],
                         mu_s=params[3],
                         vel_stribeck=params[4],
                         exp_stribeck=params[5],
                         contact_len=params[0],
                         k_bristle=params[1],
                         num_bristle=args[2])
    res_MF = magic_formula_longitudinal(vel_tyre=args[0],
                                        vel_vehicle=args[1],
                                        )
    
    return res_BB-res_MF




if __name__ == "__main__":
    # tyre parameters
    L       = 0.1                               # contact patch length          [m]         range [0.05-0.2]
    k_0     = 240                               # Bristle micro-stiffness       [1/m]       range [100-600]
    V_roll  = 16                                # Tyre rolling speed            [m/s]       range [0.1-100]
    
    # friction parameters
    mu_d    = 0.7                               # Dynamic friction coefficient  [-]         range (0,1]
    mu_s    = 1.2                               # Static friction coefficient   [-]         range [0.4,2]
    v_S     = 3.5                               # Stribeck velocity             [m/s]       range [2-10]
    delta_S = 0.6                               # Stribeck exponent             [-]         range [0.1,2]

    ## Discretization (the grid of what we vary)

    n_x     = 100                               # Spatial grid
    n_v     = 200                               # Velocity grid

    v_rel   = np.linspace(-1, 0, n_v) * V_roll      # List of all relative velocities
    v       = V_roll-v_rel                          # List of tyre velocities
    #xi      = np.linspace(0, 1, n_x) * L            # List of all spatial positions

    # Longitudinal slip
    sigma_x = -v_rel / V_roll

    initial_guess = np.array([
        0.1,
        240,
        0.7,
        1.2,
        3.5,
        0.6,
    ])

    lower_bounds = np.array([
        0.025,
        1,
        0.01,
        0.4,
        0.01,
        0.1,
    ])

    upper_bounds = np.array([
        0.25,
        10_000,
        1,
        2,
        20,
        2,
    ])

    
    res = least_squares(residual,
                        initial_guess,
                        bounds=(lower_bounds,upper_bounds),
                        args=(v, V_roll, n_x)
                        )
    
    res_2 = least_squares(residual,
                        initial_guess,
                        bounds=(lower_bounds,upper_bounds),
                        args=(v, V_roll, n_x),
                        loss="cauchy",
                        )
    
    print(res)
    
    Fs_MF = magic_formula_longitudinal(v, V_roll)
    Fs_BB = basic_brush(v, 
                        V_roll, 
                        num_bristle=n_x,
                        mu_d=res.x[2],
                        mu_s=res.x[3],
                        vel_stribeck=res.x[4],
                        exp_stribeck=res.x[5],
                        contact_len=res.x[0],
                        k_bristle=res.x[1],
                        )

    Fs_BB_2 = basic_brush(v, 
                        V_roll, 
                        num_bristle=n_x,
                        mu_d=res_2.x[2],
                        mu_s=res_2.x[3],
                        vel_stribeck=res_2.x[4],
                        exp_stribeck=res_2.x[5],
                        contact_len=res_2.x[0],
                        k_bristle=res_2.x[1],
                        )
    #print parameters
    print(res.x)
    print(res_2.x)

    residual_brush = Fs_MF-Fs_BB
    residual_brush_2 = Fs_MF-Fs_BB_2

    X = v_rel.reshape(-1,1)
    y = residual_brush
    y_2 = residual_brush_2

    sr_model = PySRRegressor(
        model_selection="best",
        maxsize=8,
        maxdepth=6,
        niterations=5000,
        populations=48,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "square", "abs"],
        turbo=True,
    )

    sr_model_2 = PySRRegressor(
        model_selection="best",
        maxsize=8,
        maxdepth=6,
        niterations=5000,
        populations=48,
        binary_operators=["+", "-", "*", "/"], #TODO remove division
        unary_operators=["exp", "square", "abs"],
        turbo=True,
    )

    sr_model.fit(X,y)
    sr_eq = str(sr_model.sympy())
    sr_model_2.fit(X,y_2)
    sr_eq_2 = str(sr_model_2.sympy())

    print(sr_model)
    print("Best equation:")
    print(sr_model.sympy())

    print(sr_model_2)
    print("Best equation:")
    print(sr_model_2.sympy())

    sr_res = sr_model.predict(X)
    sr_res_2 = sr_model_2.predict(X)
    Fs_SR = Fs_BB + sr_res
    Fs_SR_2 = Fs_BB_2 + sr_res_2

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(sigma_x, Fs_MF/1000, label="Magic Formula", linewidth=1)
    plt.plot(sigma_x, Fs_BB/1000, label="Basic Brush model", linewidth=1)
    plt.plot(sigma_x, Fs_SR/1000, label="Hybrid brush + Symbolic Regression", linewidth=1, linestyle="--")
    #plt.plot(sigma_x, Fs_SR_2/1000, label="Hybrid brush + Symbolic Regression cauchy", linewidth=1, linestyle="dashdot")
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.legend()
    plt.text(0.03,
             0.97,
             sr_eq,
             fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    #plt.text(0.03,
    #         0.93,
    #         sr_eq_2,
    #         fontsize=8,
    #         verticalalignment="top",
    #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    plt.show()
