import numpy as np
import matplotlib.pyplot as plt
from Model_library.MF_model import magic_formula_longitudinal
from Model_library.Basic_brush_model import basic_brush
from scipy.optimize import least_squares, differential_evolution
from pysr import PySRRegressor
#from sklearn.pipeline import make_pipeline, Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import r2_score, mean_squared_error
#from sklearn import linear_model


np.set_printoptions(suppress=True, precision=3)

def residual(params, *args):    
    res_BB = basic_brush(vel_roll=args[0],
                         vel_vehicle=args[1],
                         mu_d=params[2],
                         mu_s=params[3],
                         vel_stribeck=params[4],
                         exp_stribeck=params[5],
                         contact_len=params[0],
                         k_bristle=params[1],
                         num_bristle=args[2])
    res_MF = magic_formula_longitudinal(vel_roll=args[0],
                                        vel_vehicle=args[1],
                                        )
    F_ref = np.max(np.abs(res_MF))
    norm_factor = np.maximum(np.abs(res_MF), 0.05 * F_ref)
    return (res_BB-res_MF) / norm_factor




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

    #tolerance
    eps = 1e-12

    # Longitudinal slip
    sigma_x = -v_rel / V_roll

    # initial_guess = np.array([
    #     0.093,
    #     498.86,
    #     0.81,
    #     1.91*0.81,
    #     3.67,
    #     0.954,
    # ])
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

    print("starting least squares")
    res = least_squares(residual,
                        initial_guess,
                        bounds=(lower_bounds,upper_bounds),
                        args=(v, V_roll, n_x),
                        ftol=1e-12,
                        gtol=1e-12,
                        #x_scale="jac",
                        )
    
    print("starting genetic algorithm")
    genetic_bounds = list(zip(lower_bounds, upper_bounds))
    def residual_genetic(params, *args):
        r = residual(params, *args)
        return np.sum(r**2)
    res_genetic = differential_evolution(residual_genetic,
                        bounds=genetic_bounds,
                        args=(v, V_roll, n_x),
                        maxiter=10,
                        popsize=24,
                        workers=1,
                        polish=False,
                        disp=True,
                        #updating="deferred",
                        seed=42,
                        )
    
    print(res)
    
    Fs_MF = magic_formula_longitudinal(v, V_roll)
    Fs_BB, z = basic_brush(v, 
                        V_roll, 
                        num_bristle=n_x,
                        mu_d=res.x[2],
                        mu_s=res.x[3],
                        vel_stribeck=res.x[4],
                        exp_stribeck=res.x[5],
                        contact_len=res.x[0],
                        k_bristle=res.x[1],
                        return_z=True, #return z0 for symbolic term
                        )
    
    Fs_BB_genetic = basic_brush(v, 
                        V_roll, 
                        num_bristle=n_x,
                        mu_d=res_genetic.x[2],
                        mu_s=res_genetic.x[3],
                        vel_stribeck=res_genetic.x[4],
                        exp_stribeck=res_genetic.x[5],
                        contact_len=res_genetic.x[0],
                        k_bristle=res_genetic.x[1],
                        return_z=False, #return z0 for symbolic term
                        )

    #print parameters
    print("least squares results")
    print(res.x)
    print("genetic results")
    print(res_genetic.x)

    residual_brush = (Fs_MF-Fs_BB)/(res.x[0]*(z+eps))

    X = v_rel.reshape(-1,1)
    y = residual_brush

    sr_model = PySRRegressor(
        model_selection="best",
        maxsize=8,
        maxdepth=6,
        niterations=2000,
        populations=48,
        binary_operators=["+", "-", "*"],
        unary_operators=["exp", "square", "abs", "atan"],
        turbo=True,
    )

    sr_model.fit(X,y)
    sr_eq = str(sr_model.sympy())

    print(sr_model)
    print("Best equation:")
    print(sr_model.sympy())

    sr_res = sr_model.predict(X)
    Fs_SR = Fs_BB + sr_res

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(sigma_x, Fs_MF/1000, label="Magic Formula", linewidth=1)
    plt.plot(sigma_x, Fs_BB/1000, label="Basic Brush model", linewidth=1)
    plt.plot(sigma_x, Fs_BB_genetic/1000, label="Basic Brush model genetic", linewidth=1)
    plt.plot(sigma_x, Fs_SR/1000, label="Hybrid brush + Symbolic Regression", linewidth=1, linestyle="--")
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.legend()
    plt.text(0.03,
             0.97,
             str(res.x),
             fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.text(0.03,
             0.93,
             str(res_genetic.x),
             fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.text(0.03,
            0.88,
            sr_eq,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    plt.show()