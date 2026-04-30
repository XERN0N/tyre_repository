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

#show link: underline

#set document(title: "Write-up on current state")
#title()
//#outline(title: "Table of contents")

//= Overview

Currently we have coded and validated Pacejka's magic formula and a Linear brush model with stribeck friction to Luigi's implementation in Matlab and get matching results to machine precision.

//TODO
In short:
- We have re-run the least-squares fitting with different initial guesses and get conversion around XX coefficients, which fits the magic formula model well which can be seen in XX.

- We have tried Luigi's approach of coupling $mu_s$ with $mu_d$ by setting $mu_s=mu_d dot k_s$ where $k_s$ is fitted during the least squares fit instead of $mu_s$ directly. The results can be seen in XX.

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
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, left, center, center),
    table.header[Parameter][Symbol][Units][Physical meaning][Lower bound][Upper bound],
    [Contact patch length], [$L$],         [m],     [Length of the tyre--road contact zone],   [0.05], [0.12],
    [Bristle stiffness],    [$k_0$],        [1/m],   [Lateral micro-stiffness of the bristles], [100],  [800],
    [Dynamic friction],     [$mu_d$],       [-],     [Coulomb friction at high sliding speed],  [0.7],  [2.0],
    [Static friction],      [$mu_s$],       [-],     [Friction coefficient at zero slip],        [1.0],  [3.5],
    [Stribeck velocity],    [$v_S$],        [m/s],   [Speed at friction transition midpoint],   [2],  [20.0],
    [Stribeck exponent],    [$delta_S$],    [-],     [Sharpness of the friction transition],    [0.1],  [2.0],
  ),
  caption: [Brush model parameters, their physical meaning, and the search bounds used during optimisation.],
)<param_table>

#figure(
  image("Pictures/Comparison_plots/Fitted_models/20260430_221342_Brush_model_first_fit_default.png", width: 80%),
  caption: [Comparison of the first brush model (least squares fit) against the Magic Formula reference.],
)
The residuals passed to the optimizer are normalised by $max(|F_"MF"|)$, so that all residual components are dimensionless and of order one, ensuring equal weighting across the slip range.

Using the least squares fit, means that a local minimum is searched for the cost function:

$
F(x)= frac(1,2) sum^(m-1)_(i=0) f_i (x)^2
$

Where $f_i$ is the residual. The method used for finding the local minimum is called trust region reflective algorithm, an initial guess is given with boundaries. The initial guess is evaluated, giving the current residual vector and the current loss. A jacobian is then found by perturbating the parameters in our current guess. this creates in our case a 200x6 matrix, where 200 is because we have descitized the slip ratio, and 6 because we have 6 parameters.
The jacobian is used to build a local approximation, where the residual $r(x_k+p) approx r(x_k)+J dot p$, where $p$ is the proposed parameter step, and and $x_k+p$ is a proposed parameter vector.
Using this residual, a local approximation is build:
$m_k (p) = frac(1,2)||r(x_k)+J dot p||^2$. The approximation is only reliable near our currently parameter vector $x_k$, since the jacobian is found by perturbation. Thus we define a trusted region $ || p|| <= Delta $. This also complies with the bounds that is given, meaning $l <= x_k+p <= u$. SciPy then tries to find the best step p, such that the approximate loss is reduced
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

A key weakness of this approach is sensitivity to the initial guess: because the trust region only explores a local neighbourhood, the optimizer can converge to a suboptimal local minimum if the starting point is far from the global one.

== Genetic algorithm (differential evolution) <genetic>

Differential evolution is a population-based global search algorithm that does not require a gradient or a good initial guess. Instead of following a single parameter vector, it maintains a population of candidate solutions (in this case $12 times 6 = 72$ candidates) that are evolved over up to 200 generations. At each generation, trial vectors are formed by combining randomly selected members of the population; a trial vector replaces its parent only if it produces a lower cost. Because the entire parameter space bounded by the search bounds is explored, the algorithm is far less likely to get stuck in a local minimum than the trust region method.

The main trade-off is computational cost: the genetic optimizer evaluates the residual function many more times than least squares. Parallelism over all available CPU cores (via Python multiprocessing) is used to mitigate this. A fixed random seed (42) is set for reproducibility.

Comparing the two optimizers on the same problem makes it possible to judge whether the least-squares solution is a true global minimum (both methods agree) or merely a local one (the results differ).

== Basic brush model using coupled friction least squares fit
#figure(
  image("Pictures/Comparison_plots/Sanity_check_300iter_100pop/Comparison.png", width: 60%),
  caption: [Comparison of the brush model (coupled friction, least squares fit) against the Magic Formula reference.],
)

The coupled friction refers to the coupling of the static and dynamic friction coefficients. Originally these two parameters were independent, and were only related through the expression in @fric. The problem with independent bounds is that the optimizer could set $mu_s <= mu_d$, which is physically incorrect: static friction must always exceed dynamic friction.

By introducing the coupling $mu_d = k_d$ and $mu_s = k_s dot mu_d$ with the constraint $k_s > 1$ enforced through the parameter bounds, the physical ordering $mu_s > mu_d$ is guaranteed for every candidate solution the optimizer evaluates. In practice this also expands the effective search range for $mu_s$, since it is no longer capped by a fixed upper bound but scales with $mu_d$.


== Brush model performance comparison
With Pacejka's magic formula from @MF_Guiggiani as the benchmark to match, the brush model with coefficients obtained by both optimizers is shown in @comparison_fig.
#figure(
  image("Pictures/Comparison_plots/Baseline_plots_28_04_2026/Comparison_MF_BB_Genetic.png", width: 100%),
  caption: [Lateral force curves for the Magic Formula, the least-squares-fitted brush model, and the genetic-algorithm-fitted brush model.],
)<comparison_fig>

Both optimizers reproduce the linear (low-slip) region well, which is governed primarily by the product $k_0 dot L$ and is relatively easy to match. The main differences appear near the saturation region: the brush model with a Stribeck friction curve is structurally limited in how sharply it can peak and drop off compared to the sinusoidal shape of the Magic Formula. The genetic optimizer, exploring the full parameter space, generally achieves a lower final cost than least squares when the initial guess is far from the global minimum.


#colbreak()
== Discussion

The brush model is a physics-based alternative to the empirical Magic Formula. Its parameters each carry a physical interpretation (@param_table), which makes it easier to reason about how tyre behaviour changes with load or surface conditions. However, fitting the model to an existing Magic Formula curve reveals its structural limitations: the Stribeck friction model introduces a smooth exponential decay that cannot exactly replicate the sharp saturation shape of the Magic Formula for all parameter choices.

The coupled friction formulation (Section 4) is an important improvement over the uncoupled version, as it enforces a physically meaningful constraint ($mu_s > mu_d$) and avoids degenerate solutions where the optimizer exploits the independence of the two bounds.

Future work could extend the model to combined longitudinal and lateral slip (the full 3D brush model), compare against real experimental tyre data rather than a reference formula, or explore alternative friction models such as the LuGre model for improved accuracy in the saturation region.

#bibliography("references.bib")
