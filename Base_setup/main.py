import numpy as np
import matplotlib.pyplot as plt
from Model_library.MF_model import magic_formula_longitudinal
from Model_library.Basic_brush_model import basic_brush
from scipy.optimize import least_squares, differential_evolution
from pysr import PySRRegressor

np.set_printoptions(suppress=True, precision=3)

def residual(params, *args):    
    res_BB = basic_brush(contact_len=params[0],
                         k_bristle=params[1],
                         mu_d=params[2],
                         mu_s=params[3],
                         vel_stribeck=params[4],
                         exp_stribeck=params[5],
                         v_rel=args[0],
                         v_tyre=args[1],
                         num_bristle=args[2],
                         load_fz=args[3])
    res_MF = magic_formula_longitudinal(v_rel=args[0],
                                        v_tyre=args[1],
                                        load_fz=args[3],
                                        )
    # F_ref = np.max(np.abs(res_MF))
    # norm_factor = np.maximum(np.abs(res_MF), 0.05 * F_ref)
    return (res_BB-res_MF)# / norm_factor




if __name__ == "__main__":
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
    xi      = np.linspace(0, 1, n_x) * L         # List of all spatial positions

    # Longitudinal slip
    sigma_x = -v_rel / v_tyre

    # initial_guess = np.array([
    #     0.093,
    #     498.86,
    #     0.81,
    #     1.91*0.81,
    #     3.67,
    #     0.954,
    # ])
    initial_guess = np.array([
        L,
        k_0,
        mu_d,
        mu_s,
        v_S,
        delta_S,
    ])

    lower_bounds = np.array([
        0.05,
        100,
        0.7,
        1,
        0.1,
        0.1,
    ])

    upper_bounds = np.array([
        0.12,
        800,
        1.2,
        2,
        20,
        2,
    ])

    print("starting least squares")
    res_lstsqrs = least_squares(residual,
                        initial_guess,
                        bounds=(lower_bounds,upper_bounds),
                        args=(v_rel, v_tyre, n_x, Fz),
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
                        args=(v_rel, v_tyre, n_x, Fz),
                        maxiter=300,
                        popsize=100,
                        workers=-1,
                        polish=False,
                        disp=True,
                        updating="deferred",
                        seed=42,
                        )
    
    print(res_lstsqrs)
    
    Fs_MF = magic_formula_longitudinal(v_rel=v_rel, v_tyre=v_tyre, load_fz=Fz)
    Fs_BB = basic_brush(v_rel=v_rel, 
                        v_tyre=v_tyre, 
                        num_bristle=n_x,
                        contact_len=res_lstsqrs.x[0],
                        k_bristle=res_lstsqrs.x[1],
                        mu_d=res_lstsqrs.x[2],
                        mu_s=res_lstsqrs.x[3],
                        vel_stribeck=res_lstsqrs.x[4],
                        exp_stribeck=res_lstsqrs.x[5],
                        return_z=False, #return z0 for symbolic term
                        )
    
    Fs_BB_genetic = basic_brush(v_rel=v_rel, 
                                v_tyre=v_tyre, 
                                num_bristle=n_x,
                                contact_len=res_genetic.x[0],
                                k_bristle=res_genetic.x[1],
                                mu_d=res_genetic.x[2],
                                mu_s=res_genetic.x[3],
                                vel_stribeck=res_genetic.x[4],
                                exp_stribeck=res_genetic.x[5],
                                return_z=False, #return z0 for symbolic term
                                )

    Fs_BB_genetic_Luigi = np.genfromtxt("Base_setup/Model_library/y_data_brush_genetic_luigi.csv", dtype=float)
    genetic_luigi_parameters = np.array([
        0.1167,
        724.7433,
        1.1999,
        2.0000,
        6.6007,
        1.0509,
    ])

    #print parameters
    print("full results least squares")
    print(res_lstsqrs)
    print("full results genetic")
    print(res_genetic)
    print("-"*50)
    print("least squares results")
    print(res_lstsqrs.x)
    print("genetic results")
    print(res_genetic.x)

    # residual_brush = (Fs_MF-Fs_BB)/(res_lstsqrs.x[0]*(z+eps))

    # X = v_rel.reshape(-1,1)
    # y = residual_brush

    # sr_model = PySRRegressor(
    #     model_selection="best",
    #     maxsize=8,
    #     maxdepth=6,
    #     niterations=2000,
    #     populations=48,
    #     binary_operators=["+", "-", "*"],
    #     unary_operators=["exp", "square", "abs", "atan"],
    #     turbo=True,
    # )

    # sr_model.fit(X,y)
    # sr_eq = str(sr_model.sympy())

    # print(sr_model)
    # print("Best equation:")
    # print(sr_model.sympy())

    # sr_res = sr_model.predict(X)
    # Fs_SR = Fs_BB + sr_res

    # Plot
    plt.figure(figsize=(12,8))
    plt.plot(sigma_x, Fs_MF, label="Magic Formula", linewidth=1)
    plt.plot(sigma_x, Fs_BB, label="Basic Brush model", linewidth=1)
    plt.plot(sigma_x, Fs_BB_genetic, label="Basic Brush model genetic", linewidth=1)
    plt.plot(sigma_x, Fs_BB_genetic_Luigi, label="Basic Brush model genetic Luigi", linewidth=1)
    # plt.plot(sigma_x, Fs_SR/1000, label="Hybrid brush + Symbolic Regression", linewidth=1, linestyle="--")
    plt.xlabel(r'Longitudinal slip $\sigma_x$ (-)')
    plt.ylabel(r'Longitudinal force $F_x$ (kN)')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1 * np.min(Fs/1000))
    plt.grid(True)
    plt.legend()
    plt.text(0.53,
             0.97,
             str(initial_guess)+" Initial guess parameters",
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=2))
    plt.text(0.53,
             0.91,
             str(res_lstsqrs.x)+" Least squares parameters",
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=2))
    plt.text(0.53,
             0.85,
             str(res_genetic.x)+" genetic parameters",
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=2))
    plt.text(0.53,
             0.79,
             str(genetic_luigi_parameters)+" Luigi's genetic parameters",
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=2))
    # plt.text(0.03,
    #         0.88,
    #         sr_eq,
    #         fontsize=8,
    #         verticalalignment="top",
    #         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    plt.savefig("Comparison.png")
    plt.show()