import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from Model_library.MF_model import magic_formula_longitudinal
from Model_library.Basic_brush_model import basic_brush
from scipy.optimize import least_squares, differential_evolution
from pysr import PySRRegressor

np.set_printoptions(suppress=True, precision=3)

def residual(params, *args):    
    res_BB = basic_brush(contact_len=params[0],
                         k_bristle=params[1],
                         mu_d=params[2],
                         mu_s=params[3]*params[2], #mu_s*mu_d test
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
    import matplotlib.pyplot as plt
    import scienceplots
    #plt.style.use()
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
        2,
        3.5,
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
                        x_scale="jac",
                        )
    
    print("starting genetic algorithm")
    genetic_bounds = list(zip(lower_bounds, upper_bounds))

    def residual_genetic(params, *args):
        r = residual(params, *args)
        return np.sum(r**2)
    
    res_genetic = differential_evolution(residual_genetic,
                        bounds=genetic_bounds,
                        args=(v_rel, v_tyre, n_x, Fz),
                        maxiter=100,#300,
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
                        mu_s=res_lstsqrs.x[3]*res_lstsqrs.x[2],
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
                                mu_s=res_genetic.x[3]*res_genetic.x[2],
                                vel_stribeck=res_genetic.x[4],
                                exp_stribeck=res_genetic.x[5],
                                return_z=False, #return z0 for symbolic term
                                )

    # Fs_BB_genetic_Luigi = np.genfromtxt("Base_setup/Model_library/y_data_brush_genetic_luigi.csv", dtype=float)
    # genetic_luigi_parameters = np.array([
    #     0.1167,
    #     724.7433,
    #     1.1999,
    #     2.0000,
    #     6.6007,
    #     1.0509,
    # ])

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
    with plt.style.context(["science","no-latex", "grid", "high-vis"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sigma_x, Fs_MF,               label="Magic Formula",              linewidth=1)
        ax.plot(sigma_x, Fs_BB,               label="Basic Brush (least squares)", linewidth=1)
        ax.plot(sigma_x, Fs_BB_genetic,       label="Basic Brush (genetic)",       linewidth=1, )
        # ax.plot(sigma_x, Fs_BB_genetic_Luigi, label="Basic Brush (Luigi genetic)", linewidth=1, )
        ax.set_xlabel(r'Lateral slip $\sigma_y$ [-]')
        ax.set_ylabel(r'Lateral force normalized $F_y$ [-]')
        ax.legend(loc="right")

        col_labels = [r"$L$ [m]", r"$k_0\ \left[\frac{1}{m}\right]$", r"$\mu_d$", r"$\mu_s$", r"$v_S\ \left[\frac{m}{s}\right]$", r"$\delta_S$"]
        row_labels = ["Initial guess", "Least squares", "Genetic"] #, "Genetic (Luigi)"
        table_data = [
            [f"{v:.3f}" for v in initial_guess],
            [f"{v:.3f}" for v in res_lstsqrs.x],
            [f"{v:.3f}" for v in res_genetic.x],
            # [f"{v:.3f}" for v in genetic_luigi_parameters],
        ]

        tbl = ax.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            bbox=[0.42, 0.02, 0.56, 0.30],
            cellLoc="center",
        )
        tbl.auto_set_font_size(True)
        #tbl.set_fontsize(plt.rcParams["font.size"])
        tbl.set_zorder(10)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor("#bbbbbb")
            cell.set_facecolor("white")
            cell.set_zorder(10)

        fig.tight_layout()
        fig.savefig("Comparison.png", dpi=150)
        plt.show()