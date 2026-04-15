import sympy as sp

x, v, V_r, k0, v_eps = sp.symbols('x v V_r k0 v_eps', real=True)
mu = sp.Function('mu')
r = sp.Function('r')

z = (-v * mu(v) / ((v_eps + mu(v) * r(v)) * k0)) * (
    1 - sp.exp(-((v_eps / (mu(v) * V_r)) + r(v)) * k0 * x)
)

sp.init_printing()
sp.pprint(z)