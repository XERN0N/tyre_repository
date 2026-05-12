#import "@preview/elsearticle:1.0.0": *
//#import "@preview/mannot:0.3.1": *

//setting equation numbering
#set math.equation(numbering: "1.")
//set heading numbering
#set heading(numbering: none) //"1."
//helper function for derivatives:
#let ded(upper, lower) = math.op($(partial #upper) / (partial #lower)$)

//helper function matrix transpose
#let matr(delim: "(", ..columns) = math.mat(delim: delim, ..array.zip(..columns))

#let fmt-p(x, digits: 3) = str(calc.round(x, digits: digits))
#let fmt-c(x, digits: 2) = {
  let e = int(calc.floor(calc.log(calc.abs(x), base: 10)))
  let m = calc.round(x / calc.pow(10, float(e)), digits: digits)
  str(m) + "e" + str(e)
}
#show link: underline
#show figure.where(kind: table): set figure.caption(position: top)

#set document(title: "Write-up on current state")
#title()
//#outline(title: "Table of contents")

//= Overview

Currently we have coded and validated Pacejka's magic formula and a Linear brush model with stribeck friction to Luigi's implementation in Matlab and get matching results to machine precision.

//TODO
In short:
- We have re-run the least-squares fitting with different initial guesses and get results that fit the magic formula model well which can be seen in @First_fit.

- We have tried Luigi's approach of scaling $mu_s$ with $mu_d$ by setting $mu_s=mu_d dot k_s$ where $k_s$ is fitted during the least squares fit instead of $mu_s$ directly. The results can be seen in @Scaled_brush.

- The least squares solver generally converged to a good optimum close to that of the genetic algorithm and the stribeck frictional model can obtain similar characteristics as the magic formula by Pacejka.

= Pacejka's magic formula
The magic formula model (lateral) used has coefficients taken from #cite(<Guiggiani>) equation 2.114:
$
F_y (sigma_y)=-1100 dot sin{1.48 dot arctan[12.27 dot sigma_y -0.07(12.27 dot sigma_y-arctan (12.27 dot sigma_y))]}
$<MF_Guiggiani>

#figure(
  image("Pictures/Comparison_plots/Baseline_plots_28_04_2026/MF_baseline.png", width: 60%),
  caption: [Lateral force predicted by Pacejka's Magic Formula (@MF_Guiggiani[]) for an FSAE tyre under pure lateral slip.],
)
This is the equation for the lateral force $F_y$ with lateral slip ratio $sigma_y$. The coefficients B, C, D and E are written out explicitly: B = 12.27 is the stiffness factor (slope at zero slip), C = 1.48 is the shape factor, D = 1100 N is the peak force amplitude, and E = 0.07 the curvature factor. These coefficients are experimentally determined and represent the case of pure lateral slip for one specific FSAE tyre.


== Linear brush model using least squares fit
The brush model's parameters are fitted using SciPy's non-linear least squares implementation (#link("https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html")[link]) with finite difference gradients.

For the brush model we are using the brush model solution in the form of:

$       //saved bristle deflection formula
  z(x) =  -frac("sgn"_epsilon (v) mu(v) , overline(k_0)) (1-exp(-frac(|v|_epsilon overline(k_0),mu(v) V_r)x)), #h(1cm) "where" x in (0,L)
$<z_deflection>
With the frictional coefficient modeled using Stribeck friction. 
$
mu(v)=mu_d +( mu_s - mu_d ) exp(-(frac(|v|,v_S))^(delta_S))
$<fric>

/*Where $mu_d$ is the dynamic (Coulomb) friction coefficient, $mu_s$ is the static friction coefficient, $v_S$ is the Stribeck velocity (the relative speed at which friction transitions from static to dynamic), and $delta_S$ is the Stribeck exponent (controls the sharpness of that transition).*/
By then fitting the six parameters in @param_table with the least-squares optimizer different results are obtained based on the bounds and local minima that the optimizer can land in. 

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, center, center, left, center, center, center),
    table.header([Parameter], [Symbol], [Units], [Physical meaning], [Initial \ guess], [Lower \ bound], [Upper \ bound]),
    [Contact patch length], [$L$],         [m],     [Length of the tyre--road contact zone],           [0.1],  [0.05], [0.12],
    [Bristle stiffness],    [$k_0$],        [1/m],   [Lateral micro-stiffness of the bristles],         [240],  [100],  [800],
    [Dynamic friction],     [$mu_d$],       [-],     [Friction at high slip speeds], [0.7],  [0.7],  [2.0],
    [Static friction],      [$mu_s$],       [-],     [Friction at low slip speeds],               [1.2],  [1.0],  [3.5],
    [Stribeck velocity],    [$v_S$],        [m/s],   [Characteristic speed for friction transition],   [3.5],  [2],    [20.0],
    [Stribeck exponent],    [$delta_S$],    [-],     [Sharpness of  friction transition],            [0.6],  [0.1],  [2.0],
  ),
  caption: [Brush model parameters for the first fit and the search bounds used during optimisation.],
)<param_table>

The residuals passed to the optimizer are normalised by $max(|F_"MF"|)$, so that all residual components are dimensionless and of order one, to ensure equal weighting across the slip range.

#figure(
  image("Pictures/Comparison_plots/Fitted_models/20260430_221342_Brush_model_first_fit_default.png", width: 100%),
  caption: [Comparison of the first brush model (least squares fit) against the Magic Formula reference.],
)<First_fit>

As can be seen in @First_fit the linear brush model can achieve good performance when compared to the magic formula which it was trained on. The fit using least squares was compared to the fit obtained using a genetic optimization algorithm and the local minima seems to overlap with the global minima when comparing the parameter values.

== Optimization bounds sweep
We also tried sweeping over some of the bounds to see if this would yield different results. The bounds for $mu_s$ and $v_S$ were swept while all other parameters kept the default bounds from @param_table.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header([Symbol], [Units], [IC], [LB swept], [UB swept]),
    [$mu_s$],    [-],    [1.2], [1.0],                        [{2, 3, 4, 5, 6}],
    [$v_S$],     [m/s],  [5],   [{0.1, 0.2, 0.5, 2.0}, {2.0}],      [{8}, {8, 12, 16, 20}],
  ),
  caption: [Parameters and bound ranges used in the bounds sweep totalling 40 parameter combination. All other parameters held at defaults from @param_table.],
)
The results were sorted by cost (rank) then filtered to only show results with parameters that differed by at least 10%.
#let bounds_diff = json("Datasets/Bound_sweep_first_fit/parameter_diff.json")

#figure(
  table(
    columns: (auto, auto, auto, ..bounds_diff.param_names.map(_ => auto)),
    align: center,
    table.header(
      [Rank], [Cost], [$R^2$],
      ..bounds_diff.param_names.map(n => [#n]),
    ),
    ..bounds_diff.runs.map(run => (
      [#run.rank],
      [#fmt-c(run.cost, digits: 3)],
      [#if run.at("r2", default: none) != none { fmt-p(run.r2, digits: 4) } else { [-] }],
      ..bounds_diff.param_names.map(n => [#fmt-p(run.params.at(n), digits: 3)]),
    )).flatten(),
  ),
  caption: [Distinct parameter sets from the bounds sweep, ranked by fit cost.],
)<First_fit_bounds>

As can be seen in @First_fit_bounds the optimizer either hits upper bounds on $mu_s$ or lower bounds on $v_S$ to get the best fit. This may be due to the magic formula not distinguishing between the relative speed between the surfaces as only the slip ratio is used and not the relative speed between the surfaces. The stribeck frictional model can provide different friction for slips at speeds so that a 50% slip at 10 m/s is different to 50% slip at 100 m/s.

== Initial guess sweep
We also tried varying the initial guesses to see if the
starting guess changed the minima that the optimizer landed in.

=== Contact length and bristle stiffness
The first initial guesses to sweep over are the contact length and bristle stiffness. They are chosen as there may exist different optima with a shorter contact patch and stiffer bristles versus a longer contact patch and softer bristles. The Swept initial conditions for the first initial condition sweep are shown in @First_fit_IC_1.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    table.header([Symbol], [Units], [Base IC], [Values swept]),
    [$L$],    [m],    [0.1],  [{0.05, 0.09, 0.12}],
    [$k_0$],  [1/m],  [240],  [{100, 200, 300, 400, 500, 600, 700, 800}],
  ),
  caption: [Parameters varied in the initial guess sweep. All other parameters held at base IC from @param_table.],
)<First_fit_IC_1>

The results were sorted by cost (rank) then filtered to only show results with parameters that differed by at least 10%.

#let IC_diff = json("Datasets/Initial_condition_first_fit/20260503_200641_IC_sweep_L_005_012_k0_100_800_rerun/ic_rerun_L_k0.json")
#figure(
  table(
    columns: (auto, auto, auto, auto, ..IC_diff.param_names.map(_ => auto)),
    align: (center, left, center, center, ..IC_diff.param_names.map(_ => center)),
    table.header(
      [Rank], [Starting IC], [Cost], [$R^2$],
      ..IC_diff.param_names.map(n => [#n]),
    ),
    ..IC_diff.runs.map(run => (
      [#run.rank],
      [#run.label],
      [#fmt-c(run.cost, digits: 3)],
      [#if run.at("r2", default: none) != none { fmt-p(run.r2, digits: 4) } else { [-] }],
      ..IC_diff.param_names.map(n => [#fmt-p(run.params.at(n))]),
    )).flatten(),
  ),
  caption: [Fitted parameter sets from the initial condition sweep. Starting IC shows which initial guess was used; remaining columns show where the optimizer converged.],
)<First_fit_IC_1_json>

The same cost value is reached from all initial conditions in @First_fit_IC_1_json which indicates that reaching an optimal cost value may not dependant on the initial condition of the contact length $L$ and the bristle stiffness $k_0$.

=== Frictional coefficients and Stribeck parameters

The second initial guesses to sweep over are the friction parameters ($mu_d$, $mu_s$, $v_S$, $delta_S$). These are chosen to explore whether different friction starting points lead to different local minima, since the Stribeck model has multiple parameters that interact non-linearly.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, center, center, center),
    table.header([Symbol], [Units], [Base IC], [Values swept]),
    [$mu_d$],    [-],    [0.7], [{0.7, 1.5, 2.0}],
    [$mu_s$],    [-],    [1.2], [{1.0, 2.0, 3.5}],
    [$v_S$],     [m/s],  [5],   [{0.1, 5.0, 12.0, 20.0}],
    [$delta_S$], [-],    [0.6], [{0.1, 0.75, 1.25, 2.0}],
  ),
  caption: [Parameters varied in the second initial guess sweep. All other parameters held at base IC from @param_table.],
)<Second_fit_IC_1>

The results were sorted by cost (rank) then filtered to only show results with fittedparameters that differed by at least 10%.

#let IC_diff2 = json("Datasets/Initial_condition_first_fit/20260503_201805_IC_mu_d_mu_s_v_S_d_S_full_3_3_4_4/mu_d_mu_s_v_S_d_S_full.json")
#figure(
  table(
    columns: (auto, auto, auto, auto, ..IC_diff2.param_names.map(_ => auto)),
    align: (center, left, center, center, ..IC_diff2.param_names.map(_ => center)),
    table.header(
      [Rank], [Starting IC], [Cost], [$R^2$],
      ..IC_diff2.param_names.map(n => [#n]),
    ),
    ..IC_diff2.runs.map(run => (
      [#run.rank],
      [#run.label],
      [#fmt-c(run.cost, digits: 3)],
      [#if run.at("r2", default: none) != none { fmt-p(run.r2, digits: 4) } else { [-] }],
      ..IC_diff2.param_names.map(n => [#fmt-p(run.params.at(n))]),
    )).flatten(),
  ),
  caption: [Fitted parameter sets from the friction parameter initial condition sweep. Starting IC shows which friction initial guess was used; remaining columns show where the optimizer converged.],
)<Second_fit_IC>

The same cost value is achieved in @Second_fit_IC with the exception of start  (rank 144) that converged to a bad optimum with a cost value of $approx$ 200 times larger than the other optimum

= Linear brush model using scaled $bold(mu_s)$ friction
The scaled friction refers to replacing the independent $mu_s$ parameter with $mu_s = k_s dot mu_d$, where $k_s$ is the scale factor fitted by the optimizer. Originally these two parameters were independent, and were only related through the expression in @fric. The problem with independent bounds is that the optimizer could set $mu_s <= mu_d$.

By parametrising $mu_s = k_s dot mu_d$ with the constraint $k_s > 1$ enforced through the parameter bounds and $mu_s > mu_d$ is guaranteed for every candidate solution the optimizer evaluates. In practice this also expands the effective search range for $mu_s$, since it is no longer capped by a fixed upper bound but scales with $mu_d$.

#figure(
  image("Pictures/Comparison_plots/Scaled_friction_04_05_2026/Linear_brush_scaled.png", width: 100%),
  caption: [Comparison of the brush model (scaled $mu_s$, least squares fit) against the Magic Formula reference.],
)<Scaled_brush>

As seen in @Scaled_brush the linear brush model with scaled $mu_s$ friction shows a good fit to the magic formula with a $mu_d = 1.237$ and $mu_s = 5 dot 1.237 = 6.185$. The fit is also alike to that of the genetic algorithm in the same plot and the least squares optimizer found the same optimum. It should be noted that both the genetic algorithm and the least squares optimizer both hit the upper bound on the friction as was the case in the previous section.


== Brush model performance comparison with Luigi's code
With Pacejka's magic formula from @MF_Guiggiani as the benchmark to match, the brush model with coefficients obtained by both optimizers is shown in @comparison_fig.
#figure(
  image("Pictures/Comparison_plots/Baseline_plots_28_04_2026/Comparison_MF_BB_Genetic.png", width: 100%),
  caption: [Lateral force curves for the Magic Formula, the least-squares-fitted brush model, and the genetic-algorithm-fitted brush model.],
)<comparison_fig>

Our coded model compared to Luigi's model with Matlab's Genetic algorithm achieves the same fit for the given bounds and initial guess. 

== Least squares implementation details
Using the least squares fit, means that a local minimum is searched for the cost function:

$
F(x)= frac(1,2) sum^(m-1)_(i=0) f_i (x)^2
$

Where $f_i$ is the residual. The method used for finding the local minimum is called trust region reflective algorithm, an initial guess is given with boundaries. The initial guess is evaluated, giving the current residual vector and the current loss. A jacobian is then found by perturbating the parameters in our current guess. this creates in our case a 200x6 matrix, where 200 is because we have descitized the slip ratio, and 6 because we have 6 parameters.
The jacobian is used to build a local approximation, where the residual $r(x_k+p) approx r(x_k)+J dot p$, where $p$ is the proposed parameter step, and and $x_k+p$ is a proposed parameter vector.
Using this residual, a local approximation is build:
$m_k (p) = frac(1,2)||r(x_k)+J dot p||^2$. The approximation is only reliable near our currently parameter vector $x_k$, since the jacobian is found by perturbation. Thus we define a trusted region.
#nonumeq($ ||p|| <= Delta $)
 This also complies with the bounds that is given, meaning $l <= x_k+p <= u$. SciPy then tries to find the best step p, such that the approximate loss is reduced
$
op("min",
     limits: #true)_(p) frac(1,2)||r(x_k)+J dot p ||^2
     $
for which it will then propose $x_"trail"=x_k+p$ with this $p$ and $x_"trail"$, the predicted improvement is evaluated
$
rho=frac(m_k (0)- m_k (p),F(x_k)-F(x_"trail"))
$
If this ratio is good, it means that the local approximation predicts the real behavior well an will then take $x_(k+1) = x_"trail"$, if the ratio is bad then $x_(k+1)=x_k$.
now SciPy updates the trust region $Delta$, where if the prediction was good, $Delta$ will grow, thus letting the model take larger steps. If the prediction was bad, $Delta$ decreases and the model will be more cautious.
If a parameter is close to a bound, and the proposed change would violate this bound, the model will either reject the step, and reduce the trust region. Reduce the step for that specific parameter. Or try to change the step in the other parameters to compensate for the lack of step of the limited parameter.
This is the full algorithm that will be repeated until it converges and a criteria is met.

/*
A key weakness of this approach is sensitivity to the initial guess: because the trust region only explores a local neighbourhood, the optimizer can converge to a suboptimal local minimum if the starting point is far from the global one.
*/
/*
== Genetic algorithm (differential evolution) <genetic>

Differential evolution is a population-based global search algorithm that does not require a gradient or a good initial guess. Instead of following a single parameter vector, it maintains a population of candidate solutions (in this case $12 times 6 = 72$ candidates) that are evolved over up to 200 generations. At each generation, trial vectors are formed by combining randomly selected members of the population; a trial vector replaces its parent only if it produces a lower cost. Because the entire parameter space bounded by the search bounds is explored, the algorithm is far less likely to get stuck in a local minimum than the trust region method.

The main trade-off is computational cost: the genetic optimizer evaluates the residual function many more times than least squares. Parallelism over all available CPU cores (via Python multiprocessing) is used to mitigate this. A fixed random seed (42) is set for reproducibility.

Comparing the two optimizers on the same problem makes it possible to judge whether the least-squares solution is a true global minimum (both methods agree) or merely a local one (the results differ).
*/

#bibliography("references.bib")

/*
#colbreak()
== Discussion

The brush model is a physics-based alternative to the empirical Magic Formula. Its parameters each carry a physical interpretation (@param_table), which makes it easier to reason about how tyre behaviour changes with load or surface conditions. However, fitting the model to an existing Magic Formula curve reveals its structural limitations: the Stribeck friction model introduces a smooth exponential decay that cannot exactly replicate the sharp saturation shape of the Magic Formula for all parameter choices.

The coupled friction formulation (Section 4) is an important improvement over the uncoupled version, as it enforces a physically meaningful constraint ($mu_s > mu_d$) and avoids degenerate solutions where the optimizer exploits the independence of the two bounds.

Future work could extend the model to combined longitudinal and lateral slip (the full 3D brush model), compare against real experimental tyre data rather than a reference formula, or explore alternative friction models such as the LuGre model for improved accuracy in the saturation region. */

