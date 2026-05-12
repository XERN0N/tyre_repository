# Explanation: Two-dimensional FrBD Friction Models for Rolling Contact with Viscoelasticity

**Target audience:** No assumed prior knowledge beyond basic physics and calculus.

---

## Table of Contents

1. [The Big Picture: What is this paper about?](#1-the-big-picture)
2. [Rolling Contact: When Things Roll Over Each Other](#2-rolling-contact)
3. [Friction: The Basics](#3-friction-the-basics)
4. [The Bristle Model: Imagining Tiny Hairs](#4-the-bristle-model)
5. [Viscoelasticity: Materials with Memory](#5-viscoelasticity)
6. [Spring-Dashpot Models (Rheological Models)](#6-spring-dashpot-models)
7. [The Generalised Maxwell (GM) and Kelvin-Voigt (GKV) Models](#7-gm-and-gkv-models)
8. [The FrBD Friction Model: Equations](#8-the-frbd-friction-model)
9. [From Lumped to Distributed: Going to PDEs](#9-from-lumped-to-distributed)
10. [Mathematical Properties: Well-posedness and Passivity](#10-mathematical-properties)
11. [Three Model Variants (Models 1, 2, 3)](#11-three-model-variants)
12. [Numerical Results: What the Simulations Show](#12-numerical-results)
13. [Summary and Key Takeaways](#13-summary)

---

## 1. The Big Picture

This paper is about **how to model friction when something rolls** — specifically when the material is **viscoelastic** (like rubber in a tyre).

Think of a car tyre rolling on a road. The tyre exerts a force on the road (friction), which is what propels the car forward when you accelerate or slows it down when you brake. Getting this friction force right is critical for vehicle safety and control.

The challenge: rubber is not a simple material. When you deform rubber, it doesn't instantly spring back — it "remembers" being deformed for a while. This time-dependent behaviour is called **viscoelasticity**, and it makes the friction force harder to predict.

The paper's contribution is a rigorous mathematical model — called **FrBD** (Friction with Bristle Dynamics) — that:
- Works in **two dimensions** (longitudinal and lateral forces)
- Handles **viscoelastic materials** with multiple timescales of memory
- Is described by **partial differential equations** (PDEs) over the contact area
- Is proven to be mathematically sound (**well-posed** and **passive**)

---

## 2. Rolling Contact

When a ball or cylinder rolls over a surface, the two bodies touch in a **contact area** (also called contact patch). For a tyre on a road, this is roughly the size of your hand.

### Coordinate system

The paper uses a coordinate frame $(O; x, y, z)$ fixed to the contact area:
- $x$: longitudinal — pointing in the rolling direction
- $y$: lateral — pointing sideways
- $z$: vertical — pointing downward into the surface

### Slip

When two bodies roll together, they can have **relative motion** between their surfaces even if they're rolling (not purely sliding). This relative motion is called **slip** or **creepage**.

**Translational slip** $\boldsymbol{\sigma}(s) = [\sigma_x(s),\, \sigma_y(s)]^\top$ describes how much the surfaces slide relative to each other:
- $\sigma_x$: longitudinal creep (e.g., when braking or accelerating)
- $\sigma_y$: lateral creep (e.g., when cornering)

**Spin slip** $\varphi(s)$ describes rotation around the vertical axis — important at high camber angles or during parking.

### The "travelled distance" variable $s$

Instead of time $t$, the paper often uses **travelled distance**:
$$s = \int_0^t V_r(t')\, dt'$$
where $V_r(t)$ is the rolling speed. This is convenient because the physics of rolling contact depends on how far the wheel has rolled, not directly on time.

---

## 3. Friction: The Basics

### Coulomb-Amontons friction law

The simplest model of friction says the friction force $F$ is proportional to the normal force $N$:
$$F = \mu N$$
where $\mu$ is the **friction coefficient**. This is the Coulomb-Amontons law taught in introductory physics.

For a rubber tyre, $\mu$ is not constant — it depends on the **sliding velocity** $v_s$ between the surfaces. At low sliding speed, friction is high (static friction), and it drops to a lower value at high sliding speed (dynamic friction). This is called the **Stribeck effect**:

$$\mu(v_s) = \mu_d + (\mu_s - \mu_d)\exp\!\left(-\!\left(\frac{\|v_s\|_2}{v_S}\right)^{\delta_S}\right) + \mu_v(v_s)$$

where:
- $\mu_s$: static friction coefficient (high, when barely sliding)
- $\mu_d$: dynamic friction coefficient (lower, when sliding fast)
- $v_S$: Stribeck velocity (the speed where friction starts dropping)
- $\delta_S$: shape parameter
- $\mu_v$: viscous friction term

### Friction in 2D

In two dimensions, friction acts in both $x$ and $y$ directions. The paper uses a **matrix** of friction coefficients:
$$\mathbf{M}(v_s) = \begin{bmatrix} \mu_{xx}(v_s) & \mu_{xy}(v_s) \\ \mu_{xy}(v_s) & \mu_{yy}(v_s) \end{bmatrix}$$

This matrix is **symmetric and positive definite** — meaning friction always opposes motion and always dissipates energy. For isotropic friction (same in all directions), $\mathbf{M}(v_s) = \mu(v_s)\mathbf{I}_2$.

The friction force (per unit normal load) acting on the bristle tip is:
$$\boldsymbol{f}_r(\boldsymbol{v}_s) = -\frac{\mathbf{M}^2(\boldsymbol{v}_s)\boldsymbol{v}_s}{\|\mathbf{M}(\boldsymbol{v}_s)\boldsymbol{v}_s\|_{2,\varepsilon}}$$

This formula ensures:
- Force points **opposite** to sliding velocity (hence the minus sign)
- Magnitude is bounded by the friction law
- The $\varepsilon$ regularisation avoids division by zero when $v_s = 0$

---

## 4. The Bristle Model

### Intuition

Imagine the contact surface is covered with millions of tiny elastic hairs — **bristles**. When the upper body slides over the lower body, these bristles bend. The bent bristles exert a restoring force that opposes the sliding motion. This is the **bristle model** of friction.

Physically, these bristles represent microscopic asperities (roughness peaks) on the surfaces. When the bodies slide, asperities deflect and eventually snap back.

### Bristle deflection $\boldsymbol{z}$

The bristle deflection is a 2D vector:
$$\boldsymbol{z} = [z_x,\, z_y]^\top$$
where $z_x$ is longitudinal deflection and $z_y$ is lateral deflection.

### Sliding velocity of the bristle tip

The tip of a bristle moves at the **sliding velocity**:
$$\boldsymbol{v}_s(\dot{\boldsymbol{z}}, \boldsymbol{v}_r) = \boldsymbol{v}_r + \dot{\boldsymbol{z}}$$

where:
- $\boldsymbol{v}_r$: **rigid relative velocity** — how fast the two bodies move relative to each other (without considering bristle bending)
- $\dot{\boldsymbol{z}} = d\boldsymbol{z}/dt$: rate of bristle bending

So the total sliding at the tip = rigid body sliding + bending rate of bristle.

### Why not just use rigid sliding?

When bristles are sticking (not sliding at the tip), $\boldsymbol{v}_s = 0$, which means $\dot{\boldsymbol{z}} = -\boldsymbol{v}_r$ — the bristle bends to absorb the relative motion. When sliding occurs, the bristle tip slides over the surface and is limited by the friction law.

This avoids the need to explicitly track "sticking zones" and "sliding zones" — a major advantage of the bristle approach.

---

## 5. Viscoelasticity

### What is it?

Most simple models treat materials as either:
- **Elastic**: deform instantly, spring back instantly (like a rubber band in idealized form)
- **Viscous**: flow continuously under stress (like honey)

**Viscoelastic** materials behave as both, depending on how fast you deform them:
- Deform slowly → acts more viscous (flows)
- Deform quickly → acts more elastic (springs back)

Rubber is viscoelastic. When you quickly compress a rubber ball, it feels hard (elastic). When you compress it very slowly, it feels softer.

### Relaxation

If you suddenly stretch a viscoelastic material and hold it there, the stress doesn't stay constant — it **relaxes** over time. This is called **stress relaxation**.

The time it takes to relax is the **relaxation time** $\tau$. Simple materials have one relaxation time. Real rubber has **many relaxation times** spanning many orders of magnitude (from milliseconds to minutes).

### Why does this matter for friction?

In a rolling tyre, the rubber in the contact patch is constantly being compressed and released as the tyre rolls. Different parts of the rubber relax at different rates. If your model only has one relaxation time, you miss dynamics at other frequencies. This is why the paper uses **Generalised** models with $n$ relaxation times.

### Energy dissipation = hysteresis

When you stretch and release a viscoelastic material, it takes more energy to stretch it than you get back when it releases. This **energy loss** (dissipated as heat) is called **hysteresis**. For rubber tyres, hysteresis friction is one of the main friction mechanisms — the rubber grips by deforming and dissipating energy.

---

## 6. Spring-Dashpot Models

Viscoelastic behaviour is represented using combinations of two basic elements:

### The Spring (elastic element)

A spring stores energy elastically. Force $f$ is proportional to displacement $z$:
$$f = k\, z$$
where $k$ is the spring stiffness. The spring responds **instantaneously** — pull it and it pulls back immediately.

In the paper, stiffness is written as a **matrix** $\bar{\mathbf{K}}$ (to handle 2D coupling):
$$\boldsymbol{f} = \bar{\mathbf{K}}\,\boldsymbol{z}$$

### The Dashpot (viscous element)

A dashpot (think of a shock absorber) dissipates energy. Force is proportional to velocity:
$$f = c\, \dot{z}$$
where $c$ is the damping coefficient and $\dot{z} = dz/dt$ is the rate of deformation.

In matrix form:
$$\boldsymbol{f} = \bar{\mathbf{C}}\,\dot{\boldsymbol{z}}$$

### Kelvin-Voigt (KV) element

Connect a spring and dashpot **in parallel**:

```
Spring k
━━━━━━━┫
        ┣━━
━━━━━━━┫
Dashpot c
```

Both elements share the same displacement $z$. Their forces add:
$$\boldsymbol{f} = \bar{\mathbf{K}}\,\boldsymbol{z} + \bar{\mathbf{C}}\,\dot{\boldsymbol{z}}$$

**Physical intuition:** Pull slowly → spring dominates. Pull fast → dashpot provides extra resistance.

### Maxwell element

Connect a spring and dashpot **in series**:

```
━━━ Spring k ━━━ Dashpot c ━━━
```

The same force passes through both. The total displacement rate is:
$$\dot{z} = \frac{\dot{f}}{k} + \frac{f}{c}$$

Or rearranged as an ODE:
$$\dot{f} = -\frac{1}{\tau}f + k\,\dot{z}, \quad \tau = \frac{c}{k}$$

where $\tau = c/k$ is the **relaxation time**. After a sudden stretch, the stress in a Maxwell element decays exponentially:
$$f(t) = f_0 e^{-t/\tau}$$

This is exactly **stress relaxation** — the fundamental signature of viscoelasticity.

---

## 7. GM and GKV Models

### Generalised Maxwell (GM) Model

To capture **multiple relaxation times**, put $n$ Maxwell elements **in parallel** with one additional spring:

```
Spring K̄₀
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spring K̄₁ ━ Dashpot C̄₁
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Spring K̄₂ ━ Dashpot C̄₂
━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ...
Spring K̄ₙ ━ Dashpot C̄ₙ
━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Each branch $i$ has its own relaxation time $\tau_i = \bar{C}_i\bar{K}_i^{-1}$ (in matrix form). The total force is:

$$\boldsymbol{f} = \bar{\mathbf{K}}_0\boldsymbol{z} + \sum_{i=1}^n \boldsymbol{f}_i \tag{GM-a}$$

where each internal force $\boldsymbol{f}_i$ satisfies the ODE:

$$\frac{d\boldsymbol{f}_i}{dt} = -\boldsymbol{\tau}_i^{-1}\boldsymbol{f}_i + \bar{\mathbf{K}}_i\dot{\boldsymbol{z}}, \quad i \in \{1,\ldots,n\} \tag{GM-b}$$

**Understanding GM-b:**
- $\boldsymbol{\tau}_i^{-1}\boldsymbol{f}_i$: this term causes $\boldsymbol{f}_i$ to decay toward zero exponentially (relaxation)
- $\bar{\mathbf{K}}_i\dot{\boldsymbol{z}}$: this term drives $\boldsymbol{f}_i$ to increase when bristle is bending

Each branch has its own timescale $\tau_i$. Fast branches ($\tau_i$ small) respond quickly and then relax quickly. Slow branches ($\tau_i$ large) build up slowly and hold their force longer.

The **relaxation time matrices** are:
$$\boldsymbol{\tau}_i \triangleq \bar{\mathbf{C}}_i\bar{\mathbf{K}}_i^{-1}$$

In the diagonal (isotropic) case:
$$\bar{\mathbf{K}}_i = \begin{bmatrix} \bar{k}_{ix} & 0 \\ 0 & \bar{k}_{iy} \end{bmatrix}, \quad \boldsymbol{\tau}_i = \begin{bmatrix} \tau_{ix} & 0 \\ 0 & \tau_{iy} \end{bmatrix}$$

The off-diagonal terms in the full matrices model **coupling** between longitudinal and lateral directions — important for anisotropic materials like oriented polymer chains.

### Generalised Kelvin-Voigt (GKV) Model

Instead of Maxwell elements in parallel, put $n$ Kelvin-Voigt elements **in series** with a main spring:

```
━━━ K̄₀ ━━━[K̄₁ ∥ C̄₁]━━━[K̄₂ ∥ C̄₂]━━━...━━━[K̄ₙ ∥ C̄ₙ]━━━
```

Now the state variables are internal **deformations** $\boldsymbol{z}_i$ (not forces). The relationships are:

$$\boldsymbol{f} = \bar{\mathbf{K}}_0\boldsymbol{z} - \bar{\mathbf{K}}_0\sum_{i=1}^n \boldsymbol{z}_i \tag{GKV-a}$$

$$\boldsymbol{f} = \bar{\mathbf{K}}_i\boldsymbol{z}_i + \bar{\mathbf{C}}_i\dot{\boldsymbol{z}}_i, \quad i \in \{1,\ldots,n\} \tag{GKV-b}$$

**Understanding GKV:**
- The total deformation $\boldsymbol{z}$ is shared across the branches in series
- Each branch "absorbs" some of the deformation: $\boldsymbol{z}_i$ is the deformation of branch $i$
- The "effective" deformation driving the main spring is $\boldsymbol{z} - \sum_i \boldsymbol{z}_i$

### Equivalence of GM and GKV

Here's a key insight: **GM and GKV are mathematically equivalent** for the same order $n$. Any GM model can be rewritten as a GKV model and vice versa. They represent two different but equivalent ways of expressing the same viscoelastic constitutive relationship:

$$\boldsymbol{f} + \sum_{i=1}^n \boldsymbol{\Gamma}_i \frac{d^i \boldsymbol{f}}{dt^i} = \sum_{i=0}^n \boldsymbol{\Sigma}_i \frac{d^i \boldsymbol{z}}{dt^i}$$

This is the **general differential constitutive equation** (Eq. 10 in the paper) relating force to deformation for any linear viscoelastic solid.

---

## 8. The FrBD Friction Model

### The core idea

The FrBD model combines:
1. A viscoelastic bristle (GM or GKV rheology) that deforms under contact
2. A friction law that limits the force when sliding

At equilibrium (no acceleration of bristle), the bristle force must equal the friction force:
$$\underbrace{\boldsymbol{f}}_{\text{bristle force}} = \underbrace{\boldsymbol{f}_r(\boldsymbol{v}_s)}_{\text{friction force}}$$

From this balance condition, the equation of motion for the bristle is derived.

### The main ODE (FrBD$_{n+1}$-GM)

After mathematical manipulation (applying the Implicit Function Theorem — a way to solve implicit equations), the dynamics of the bristle in the GM formulation are:

$$\dot{\boldsymbol{z}}(t) = -\mathbf{M}^{-2}\!\bigl(\boldsymbol{v}_r(t)\bigr)\,\bigl\|\mathbf{M}\bigl(\boldsymbol{v}_r(t)\bigr)\boldsymbol{v}_r(t)\bigr\|_{2,\varepsilon} \left(\bar{\mathbf{K}}_0\boldsymbol{z}(t) + \sum_{i=1}^n \boldsymbol{f}_i(t)\right) - \boldsymbol{v}_r(t) \tag{19a}$$

$$\dot{\boldsymbol{f}}_i(t) = -\boldsymbol{\tau}_i^{-1}\boldsymbol{f}_i(t) + \bar{\mathbf{K}}_i\dot{\boldsymbol{z}}(t), \quad i \in \{1,\ldots,n\} \tag{19b}$$

**Breaking down Eq. 19a word by word:**

- $\dot{\boldsymbol{z}}$: rate of bristle deflection (what we're solving for)
- $-\boldsymbol{v}_r(t)$: the rigid body velocity tries to push the bristle
- $\mathbf{M}^{-2}(\cdot)\|\cdots\|_{2,\varepsilon}$: a normalizing factor related to the friction coefficient
- $(\bar{\mathbf{K}}_0\boldsymbol{z} + \sum_i \boldsymbol{f}_i)$: total current bristle force
- The whole first term: friction force acts to pull the bristle back when it's over-extended

**Physical interpretation:** The bristle bends to accommodate the relative body motion. If the bristle gets too bent, the friction force at its tip pulls it back. The viscoelastic branches $\boldsymbol{f}_i$ add "memory" — they slow down the response and add time-dependent relaxation.

### Steady state

At steady state (constant velocity $\boldsymbol{v}_r$, derivatives are zero), both GM and GKV give the same friction force:
$$\boldsymbol{f}(\boldsymbol{v}_r) = -\frac{\mathbf{M}^2(\boldsymbol{v}_r)\boldsymbol{v}_r}{\|\mathbf{M}(\boldsymbol{v}_r)\boldsymbol{v}_r\|_{2,\varepsilon}} \tag{21}$$

This is just the static friction law — the viscoelastic branches $\boldsymbol{f}_i$ all go to zero at steady state for the GM model. The viscoelastic effects only matter during **transients** (when velocity is changing).

However — and this is important — once you extend to a **distributed** model over the contact area, viscoelastic effects appear even in steady state because different parts of the contact area are at different stages of their relaxation as the material flows through the contact patch.

---

## 9. From Lumped to Distributed (PDEs)

### Why distributed?

The lumped model (ODEs) treats the entire contact area as one point. But in reality, different parts of the contact patch experience different forces. A bristle entering the leading edge is freshly undeformed, while a bristle near the trailing edge has been compressed for longer.

To capture this **spatial variation**, we need partial differential equations (PDEs).

### The Eulerian approach

Instead of tracking each bristle as it moves through the contact (which would be complicated), we fix our coordinate frame to the contact area and watch what happens at each point $\boldsymbol{x} = [x, y]^\top$ in the contact area $\mathcal{C}(s)$.

The total time derivative of any quantity $q(\boldsymbol{x}, t)$ in this moving frame becomes:
$$\frac{dq}{dt} = \frac{\partial q}{\partial t} + \bigl(\boldsymbol{V}(\boldsymbol{x},t) \cdot \nabla_{\boldsymbol{x}}\bigr)q$$

This is the **material derivative** from fluid mechanics — it says "how $q$ changes following the material" = "how $q$ changes at a fixed point" + "how $q$ changes due to material flowing past that point."

Here $\boldsymbol{V}(\boldsymbol{x},t)$ is the **transport velocity** — how fast the material surface moves through the contact frame.

### The PDE system

Substituting the Eulerian derivatives into the ODE system, and using the **travelled distance** $s$ instead of time, we get the distributed FrBD$_{n+1}$ system. In compact form (defining a state vector $\boldsymbol{u}$ stacking all bristle and internal force states):

$$\frac{\partial \boldsymbol{u}(\boldsymbol{x},s)}{\partial s} + \bigl(\bar{\boldsymbol{V}}(\boldsymbol{x},s)\cdot\nabla_{\boldsymbol{x}}\bigr)\boldsymbol{u}(\boldsymbol{x},s) = \boldsymbol{\Sigma}\bigl(\bar{\boldsymbol{v}}_r(\boldsymbol{x},s),s\bigr)\boldsymbol{u}(\boldsymbol{x},s) + \boldsymbol{h}\bigl(\bar{\boldsymbol{v}}_r(\boldsymbol{x},s)\bigr) \tag{27a}$$

**Understanding each term:**

| Term | Meaning |
|------|---------|
| $\partial \boldsymbol{u}/\partial s$ | How the state changes as the wheel rolls |
| $(\bar{\boldsymbol{V}}\cdot\nabla_{\boldsymbol{x}})\boldsymbol{u}$ | How material is transported through the contact area |
| $\boldsymbol{\Sigma}(\cdot)\,\boldsymbol{u}$ | The viscoelastic relaxation and friction forcing |
| $\boldsymbol{h}(\bar{\boldsymbol{v}}_r)$ | External driving from the rigid body velocity |

This is a **first-order hyperbolic PDE** — the same type as the wave equation, or convection equations in fluid dynamics. Information travels along **characteristics** — pathlines of the material through the contact zone.

### Boundary condition: the leading edge

We must specify $\boldsymbol{u}$ where material **enters** the contact zone — at the **leading edge** $\mathcal{L}(s)$ (the front of the contact patch where material first makes contact):

$$\boldsymbol{u}(\boldsymbol{x},s) = \mathbf{0}, \quad \boldsymbol{x} \in \mathcal{L}(s) \tag{27b}$$

This says: bristles are undeformed when they enter contact. Makes physical sense — outside the contact area, there's no load.

### Transport velocity $\bar{\boldsymbol{V}}$

For a general rolling body, the nondimensional transport velocity is:

$$\bar{\boldsymbol{V}}(\boldsymbol{x},s) = -\begin{bmatrix}\varepsilon_y(s)\\ -\varepsilon_x(s)\end{bmatrix} + \mathbf{A}_{\varphi_1}(s)\boldsymbol{x}$$

where $\boldsymbol{\varepsilon}(s)$ is the rolling direction vector and $\mathbf{A}_{\varphi_1}(s)$ is a skew-symmetric matrix encoding spin:
$$\mathbf{A}_{\varphi_1}(s) = \begin{bmatrix}0 & \varphi_1(s)\\ -\varphi_1(s) & 0\end{bmatrix}$$

For simple rolling along $x$ with no spin, this reduces to $\bar{\boldsymbol{V}} \approx -[1, 0]^\top$ — material simply flows from leading to trailing edge.

### Rigid relative velocity (nondimensional) $\bar{\boldsymbol{v}}_r$

The nondimensional rigid relative velocity is:
$$\bar{\boldsymbol{v}}_r(\boldsymbol{u}(\boldsymbol{x},s), \boldsymbol{x}, s) = \bar{\boldsymbol{v}}(\boldsymbol{x},s) - \mathbf{A}_{\varphi_2}(s)\boldsymbol{z}(\boldsymbol{x},s)$$

where:
$$\bar{\boldsymbol{v}}(\boldsymbol{x},s) = -\boldsymbol{\sigma}(s) - \mathbf{A}_\varphi(s)\boldsymbol{x}$$

This captures how the local relative velocity depends on both the translational slip $\boldsymbol{\sigma}$ and the spin $\varphi$ modifying the velocity at position $\boldsymbol{x}$.

### Computing total forces

Once we solve for $\boldsymbol{u}(\boldsymbol{x},s)$, we integrate over the contact area to get total forces and moment:

$$\boldsymbol{F}_{\boldsymbol{x}}(s) = \iint_{\mathcal{C}(s)} p(\boldsymbol{x},s)\,\boldsymbol{f}(\boldsymbol{x},s)\,d\boldsymbol{x}$$

$$M_z(s) = \iint_{\mathcal{C}(s)} p(\boldsymbol{x},s)\bigl[x\,f_y(\boldsymbol{x},s) - y\,f_x(\boldsymbol{x},s)\bigr]\,d\boldsymbol{x}$$

where $p(\boldsymbol{x},s) \geq 0$ is the **pressure distribution** (how much normal force is at each point in the contact patch — typically highest in the middle, lower at the edges).

---

## 10. Mathematical Properties

### Well-posedness

Before trusting a mathematical model, we need to know that it:
1. **Has a solution** (existence)
2. **Has only one solution** (uniqueness)
3. **Is stable** — small changes in input cause small changes in solution (continuity)

These three together are called **well-posedness** (after mathematician Jacques Hadamard). If a PDE is not well-posed, numerical solutions may blow up or oscillate randomly — the model is useless.

For the FrBD PDEs, well-posedness is proved using **semigroup theory** — a mathematical framework for evolution equations. The key result (Theorems 4.1 and 4.2 in the paper): given smooth enough inputs and a bounded contact area with piecewise smooth boundary, the PDE has a unique solution that lies in appropriate function spaces ($L^2$ or $H^1$).

### What is Passivity?

Passivity is one of the most important concepts in control theory and dynamical systems. It captures the idea that a physical system **cannot create energy** — it can only store or dissipate it.

**Intuition:** Friction always **dissipates** energy (turns kinetic energy into heat). So a valid friction model must always take energy **out** of the system — never put energy **in**. If your friction model sometimes adds energy, something is wrong — you've modelled anti-friction.

**Formal definition:** A system with input $\boldsymbol{v}$ (velocity/slip) and output $\boldsymbol{f}$ (force) is **passive** if there exists a non-negative **storage function** $W \geq 0$ (think of it as "stored energy") such that:

$$\int_0^s \underbrace{-\langle p(\cdot)\boldsymbol{f}(\cdot,s'), \bar{\boldsymbol{v}}(\cdot,s')\rangle}_{\text{power extracted by friction}} \,ds' \geq W\bigl(\boldsymbol{u}(\cdot,s)\bigr) - W\bigl(\boldsymbol{u}_0(\cdot)\bigr)$$

In words: **the total energy extracted from the system by friction ≥ change in stored energy**. The difference is energy dissipated.

The left side is the **supply rate** — the power delivered to the friction system (friction force × velocity = power). Passivity says this must be non-negative in a cumulative sense.

**Why does this matter?**
1. **Physical consistency:** Friction is dissipative — a passive model respects this
2. **Control design:** When you connect a passive friction model to a mechanical system (also passive), the whole system is passive → automatically stable
3. **Simulation integrity:** No spurious energy injection

### Passivity proof for FrBD$_{n+1}$-GM (Lemma 4.1)

The storage function for the GM model is:

$$W\bigl(\boldsymbol{u}(\cdot,s)\bigr) = \underbrace{\frac{1}{2}\iint_\mathcal{C} p(\boldsymbol{x})\,\boldsymbol{z}^\top(\boldsymbol{x},s)\,\bar{\mathbf{K}}_0\,\boldsymbol{z}(\boldsymbol{x},s)\,d\boldsymbol{x}}_{\text{elastic energy in main spring}} + \underbrace{\frac{1}{2}\sum_{i=1}^n\iint_\mathcal{C} p(\boldsymbol{x})\,\boldsymbol{f}_i^\top(\boldsymbol{x},s)\,\bar{\mathbf{K}}_i^{-1}\boldsymbol{f}_i(\boldsymbol{x},s)\,d\boldsymbol{x}}_{\text{energy in Maxwell branches}}$$

This is just the elastic strain energy stored in the springs, weighted by the pressure distribution. It is always non-negative because all $\bar{\mathbf{K}}_i \succ 0$ (positive definite).

The proof proceeds by:
1. Differentiating $W$ with respect to $s$ along the PDE dynamics
2. Using integration by parts (divergence theorem) to handle the spatial derivative terms
3. Applying the boundary condition $\boldsymbol{u} = 0$ at the leading edge
4. Noting that $\bar{\mathbf{K}}_i^{-1}\boldsymbol{\tau}_i^{-1} = \bar{\mathbf{C}}_i^{-1} \succeq 0$ (positive semidefinite) — so dissipation terms are always non-negative
5. Requiring the pressure condition $\nabla_{\boldsymbol{x}} \cdot p(\boldsymbol{x})\bar{\boldsymbol{V}}(\boldsymbol{x}) \leq 0$ — meaning pressure does not increase in the rolling direction (it can be constant or decreasing)

The result is:
$$\frac{dW}{ds} \leq -\langle p(\cdot)\boldsymbol{f}(\cdot,s), \bar{\boldsymbol{v}}(\cdot,s)\rangle_{L^2(\mathcal{C};\mathbb{R}^2)}$$

which is exactly the passivity inequality.

**Note on the pressure condition:** $\nabla_{\boldsymbol{x}} \cdot p(\boldsymbol{x})\bar{\boldsymbol{V}}(\boldsymbol{x}) \leq 0$ is satisfied for:
- Constant pressure (uniform contact)
- Pressure that decreases along the rolling direction — which is physically common in viscoelastic rolling (the material piles up at the leading edge)

### Input-to-State Stability (ISS)

Besides passivity, the paper also shows **ISS** — if inputs are bounded, the state remains bounded. This is important for control: it means the system won't blow up if you drive it with any physically reasonable input.

---

## 11. Three Model Variants

By choosing different levels of approximation for the transport and relative velocity, three versions of the model are obtained:

### Model 1: Standard linear FrBD$_{n+1}$ (small spin)

Simplification: assume rolling is primarily along $x$ ($\bar{\boldsymbol{V}} \approx -[1,0]^\top$) and ignore the spin effect on relative velocity.

$$\frac{\partial \boldsymbol{u}(\boldsymbol{x},s)}{\partial s} - \frac{\partial \boldsymbol{u}(\boldsymbol{x},s)}{\partial x} = \tilde{\boldsymbol{\Sigma}}(\boldsymbol{x},s)\boldsymbol{u}(\boldsymbol{x},s) + \tilde{\boldsymbol{h}}(\boldsymbol{x},s) \tag{44a}$$

The second term $-\partial \boldsymbol{u}/\partial x$ is simple **advection** — material flows in the $-x$ direction (from leading to trailing edge) at unit speed (normalized by rolling velocity).

**When to use:** Normal driving conditions with small camber angles. Wheel-rail contact, tyre road at moderate conditions.

### Model 2: Semilinear FrBD$_{n+1}$ (large spin)

Uses the exact expressions for both $\bar{\boldsymbol{V}}$ and $\bar{\boldsymbol{v}}_r$, including the spin contribution $\mathbf{A}_{\varphi_2}\boldsymbol{z}$.

$$\frac{\partial \boldsymbol{u}}{\partial s} + \bigl(\bar{\boldsymbol{V}}(\boldsymbol{x},s)\cdot\nabla_{\boldsymbol{x}}\bigr)\boldsymbol{u} = \boldsymbol{\Sigma}\bigl(\bar{\boldsymbol{v}}_r(\boldsymbol{u},\boldsymbol{x},s),s\bigr)\boldsymbol{u} + \boldsymbol{h}\bigl(\bar{\boldsymbol{v}}_r(\boldsymbol{u},\boldsymbol{x},s)\bigr) \tag{45a}$$

This is **semilinear** (the nonlinearity in $\boldsymbol{\Sigma}$ depends on $\boldsymbol{u}$ itself, making it harder to solve).

**When to use:** High camber angles, parking manoeuvres, any situation with large spin.

### Model 3: Linear FrBD$_{n+1}$ for large spin

A compromise: uses exact $\bar{\boldsymbol{V}}$ (for transport) but approximates $\bar{\boldsymbol{v}}_r \approx \bar{\boldsymbol{v}}$ (ignores the $\mathbf{A}_{\varphi_2}\boldsymbol{z}$ term in $\boldsymbol{\Sigma}$):

$$\frac{\partial \boldsymbol{u}}{\partial s} + \bigl(\bar{\boldsymbol{V}}(\boldsymbol{x},s)\cdot\nabla_{\boldsymbol{x}}\bigr)\boldsymbol{u} = \boldsymbol{\Sigma}\bigl(\bar{\boldsymbol{v}}(\boldsymbol{x},s),s\bigr)\boldsymbol{u} + \boldsymbol{h}\bigl(\bar{\boldsymbol{v}}_r(\boldsymbol{u},\boldsymbol{x},s)\bigr) \tag{46a}$$

**When to use:** Large $\varphi_1$ (transport spin) but moderate $\varphi_2$ (deformation spin). Serves as a starting point (initial iterate) for numerical solutions of Model 2.

### Summary table

| Model | Transport velocity | Relative velocity in $\boldsymbol{\Sigma}$ | Spin handling | Linearity |
|-------|-------------------|------------------------------------------|---------------|-----------|
| 1 | Simplified ($-\hat{e}_x$) | Simplified ($\bar{\boldsymbol{v}}$) | Small spin | Linear |
| 2 | Exact | Exact ($\bar{\boldsymbol{v}}_r$ depends on $\boldsymbol{z}$) | Large spin | Semilinear |
| 3 | Exact | Simplified ($\bar{\boldsymbol{v}}$) in $\boldsymbol{\Sigma}$ | Large $\varphi_1$, moderate $\varphi_2$ | Linear |

---

## 12. Numerical Results

### Steady-state force-slip curves

When slip is constant, the model reaches a steady state. The **force-slip surface** maps:
- Input: slip $\boldsymbol{\sigma}$ and spin $\boldsymbol{\varphi}$
- Output: lateral force $F_y$, longitudinal force $F_x$, vertical moment $M_z$

**Key findings:**
1. For the **lumped** (non-distributed) model, all FrBD$_{n+1}$ variants give the **same steady-state force**. The viscoelastic branches relax completely at constant slip — no difference between orders.

2. For the **distributed** model, higher-order viscoelastic models **reduce** forces and moment slightly. Why? Because as material flows through the contact patch, the viscoelastic branches continuously relax spatially — they can't fully build up before the material exits the trailing edge. More branches = more relaxation = smaller forces.

3. The **vertical moment** $M_z$ is most affected by viscoelastic order. The position of its peak value shifts with increasing order.

4. At **large spin**, viscoelastic damping makes the force surfaces more symmetric — reducing the asymmetrisation caused by spin.

### Transient response (step slip input)

When slip suddenly steps from 0 to a constant value:
- **FrBD$_1$-KV** (simplest model): first-order exponential response, settles in ~0.1 m
- **FrBD$_2$-GM** and **FrBD$_3$-GM**: richer dynamics — initial overshoot, then gradual settling over ~0.3 m

**Why overshoot?** In the distributed model, relaxation time $\tau_i$ corresponds to a **relaxation length** $\ell_i = V_r \tau_i$. If the slip step occurs over a distance shorter than $\ell_i$, the slower branches haven't fully relaxed yet. They release their stored energy downstream → force overshoot.

This is physically important for tyres: rapid braking or sudden steering inputs excite these multi-timescale dynamics.

### Transient response (sinusoidal slip)

For oscillating slip at frequency $\omega$:
- Higher-order models show greater **phase lag** and **amplitude attenuation**
- This makes sense: a GM element is a **multi-pole low-pass filter** in the frequency domain
- The FrBD$_1$-KV acts as a single-pole filter; FrBD$_2$ and FrBD$_3$ have additional poles from the extra Maxwell branches

### Complex modulus

The GM element's frequency response is characterised by its **complex modulus**:

$$\mathbf{G}^*(\omega) = \mathbf{K}_0 + \sum_{i=1}^n i\omega\mathbf{C}_i(\mathbf{K}_i + i\omega\mathbf{C}_i)^{-1}\mathbf{K}_i$$

where $i = \sqrt{-1}$ (imaginary unit). The real part $\mathbf{G}'(\omega) = \text{Re}\{\mathbf{G}^*\}$ is the **storage modulus** (elastic part) and the imaginary part $\mathbf{G}''(\omega) = \text{Im}\{\mathbf{G}^*\}$ is the **loss modulus** (dissipative part).

At low frequency ($\omega \to 0$): $\mathbf{G}^* \to \mathbf{K}_0$ (only main spring matters)  
At high frequency ($\omega \to \infty$): $\mathbf{G}^* \to \mathbf{K}_0 + \sum_i \mathbf{K}_i$ (all springs matter)

This frequency dependence is precisely what a DMA (Dynamic Mechanical Analysis) experiment measures — and the paper shows how to fit $\bar{\mathbf{K}}_i$, $\bar{\mathbf{C}}_i$ to such measurements (Appendix A).

---

## 13. Summary

### What this paper does

1. **Takes existing FrBD framework** (already proved for simple Kelvin-Voigt bristle, Ref. [88])
2. **Generalises it** to arbitrary-order viscoelasticity using GM and GKV models
3. **Derives PDEs** for the distributed (spatially varying) rolling contact case
4. **Proves** that the new models are well-posed (unique solutions exist) and passive (no energy creation)
5. **Shows numerically** that higher-order viscoelasticity matters — especially for transient dynamics

### Key physical insights

| Phenomenon | Mechanism | Consequence |
|-----------|-----------|-------------|
| Multiple relaxation times | Different Maxwell branches relax at different rates | Multi-stage transient response, richer dynamics |
| Spatial relaxation | Material flows through contact patch; branches relax along the way | Viscoelastic effects appear even in steady state |
| Overshoot | Fast slip change activates slow branches, energy released downstream | Forces temporarily exceed steady state |
| Phase lag in frequency | Each branch adds a pole to the frequency response | Higher-order = more lag at high frequencies |

### Hierarchy of FrBD models (simplest to most complex)

$$\text{Dahl} \subset \text{FrBD}_2\text{-SLS} \subset \text{FrBD}_1\text{-KV} \subset \text{FrBD}_{n+1}\text{-GM/GKV}$$

Each model is a special case of the next. The paper shows all these relationships explicitly.

### Why does passivity matter for control?

If you use an FrBD model inside a vehicle controller (e.g., anti-lock braking, stability control), you need the friction model to behave physically. A passive friction model:
- Won't inject energy into the control loop
- Guarantees the closed-loop system is at least as stable as without the controller
- Simplifies formal stability proofs

The paper proves passivity for **any physically meaningful parametrisation** — you don't have to worry about accidentally choosing parameters that make the model active (energy-creating).

---

## 14. The Toy Problem: 1D Scalar Implementation

The full FrBD paper is 2D and uses Generalised Maxwell/Kelvin-Voigt rheology. To understand the **numerical implementation**, it helps to strip everything away and look at the simplest possible case: a tyre rolling in one direction, scalar bristle deflection, purely elastic bristle (no internal viscoelastic branches).

This is called the **toy problem**.

### What we're modelling

A tyre rolls along the road at speed $V_r(t) = \omega(t)R_r$ (angular velocity × tyre radius). The tyre is braking or accelerating, so there is **longitudinal slip** — the contact surface slides relative to the road.

The contact patch is a strip of length $L$ (or width $2a$ in the symmetric formulation). Along this strip, each bristle deflects by an amount $z(\xi, t)$ at position $\xi$ and time $t$.

---

### The two coordinate conventions

Two equivalent formulations appear across the documents — they use different coordinate systems for the same physics:

**Formulation A** — symmetric domain $x \in (-a, a)$:

$$\frac{\partial z(x,t)}{\partial t} - V_r(t)\frac{\partial z(x,t)}{\partial x} = -\frac{|v(x,t)|}{\mu(v(x,t))}\bar{k}_0\, z(x,t) - v(x,t), \quad (x,t) \in (-a, a)\times(0,T) \tag{A}$$

Boundary condition: $z(a, t) = 0$

**Formulation B** — one-sided domain $\xi \in (0, L)$:

$$\frac{\partial z(\xi,t)}{\partial t} + V_r(t)\frac{\partial z(\xi,t)}{\partial \xi} = -\frac{|v(\xi,t)|_\varepsilon}{\mu(v(\xi,t))}\bar{k}_0\, z(\xi,t) - v(\xi,t), \quad (\xi,t) \in (0,L)\times(0,T) \tag{B}$$

Boundary condition: $z(0, t) = 0$

**Why the sign difference?**

These are the same model with a coordinate flip. If you define $\xi = a - x$ (measuring distance from the **leading edge** inward):
- Leading edge: $x = a \iff \xi = 0$
- Trailing edge: $x = -a \iff \xi = L = 2a$

Under this substitution, $\partial/\partial x = -\partial/\partial \xi$, so the term $-V_r\,\partial z/\partial x$ becomes $+V_r\,\partial z/\partial \xi$. The sign flips, and the boundary condition moves from $x=a$ to $\xi=0$.

Formulation B is more natural for code: $\xi = 0$ is always the entry point regardless of which way the tyre rolls.

The $|\cdot|_\varepsilon$ in Formulation B is a **regularised absolute value**:
$$|v|_\varepsilon = \sqrt{v^2 + \varepsilon}$$
for small $\varepsilon > 0$. This avoids a kink (non-differentiability) at $v = 0$, which would cause numerical problems. As $\varepsilon \to 0$, it converges to $|v|$.

---

### Every term explained

Using Formulation B as the implementation basis:

$$\underbrace{\frac{\partial z}{\partial t}}_{\substack{\text{bristle deflection}\\ \text{changes over time}}} + \underbrace{V_r(t)\frac{\partial z}{\partial \xi}}_{\substack{\text{transport: material}\\ \text{flows through contact}}} = \underbrace{-\frac{|v|_\varepsilon}{\mu(v)}\bar{k}_0\, z}_{\substack{\text{friction pulls bristle back}\\ \text{(relaxation term)}}} \underbrace{- v}_{\substack{\text{slip drives}\\ \text{bristle to bend}}}$$

**Term 1: $\partial z/\partial t$**  
Rate of change of bristle deflection at a fixed point $\xi$. Zero at steady state.

**Term 2: $V_r(t)\,\partial z/\partial \xi$**  
This is the **advection** (transport) term. The material surface moves through the contact patch at speed $V_r$. From the contact-fixed frame, this looks like the bristle pattern being convected. If you froze time, the bristle deflection profile would shift downstream at speed $V_r$.

This is exactly the Eulerian description from Section 9 — the $(\boldsymbol{V}\cdot\nabla_x)z$ term, specialized to 1D with $V = V_r$.

**Term 3: $-\frac{|v|_\varepsilon}{\mu(v)}\bar{k}_0\, z$**  
Friction acts to pull the bristle back toward zero. The factor $\frac{|v|_\varepsilon}{\mu(v)}$ controls how strong this restoring force is:
- Large $|v|$ (fast sliding) → strong pull-back → bristle deflects less
- Large $\mu(v)$ (high friction coefficient) → weaker pull-back → bristle can deflect more

Dimensionally: this is a decay rate (units of 1/length when combined with $V_r$).

**Term 4: $-v$**  
The slip velocity $v$ directly drives bristle bending. When the contact surface slides, bristles bend to track the relative motion. This is the source term.

**Boundary condition: $z(0, t) = 0$**  
Bristles enter the contact undeformed. At $\xi = 0$ (leading edge), no load has been applied yet, so deflection is zero.

**Initial condition: $z(\xi, 0) = z_0(\xi)$**  
Starting bristle deflection profile. Often taken as zero if starting from rest.

---

### The slip velocity

For a tyre braking or accelerating, the relative velocity between tyre surface and road is:
$$v(t) = -V_r(t)\,\sigma_x(t)$$

where $\sigma_x(t)$ is the **theoretical longitudinal slip**:
$$\sigma_x(t) \triangleq \frac{\omega(t)R_r - v_x(t)}{\omega(t)R_r}$$

Here:
- $\omega(t)$: wheel angular velocity (rad/s)
- $R_r$: tyre rolling radius (m)
- $v_x(t)$: vehicle longitudinal velocity (m/s)

**Physical meaning of $\sigma_x$:**
- $\sigma_x = 0$: free rolling (no braking, no driving) — tyre surface speed equals vehicle speed
- $\sigma_x = 1$: locked wheel ($\omega = 0$, pure sliding)
- $\sigma_x > 0$: braking (tyre slower than vehicle)
- $\sigma_x < 0$: driving (tyre faster than vehicle)

The minus sign in $v = -V_r\sigma_x$ means:
- Braking ($\sigma_x > 0$) → $v < 0$ (tyre surface moves backward relative to road)
- Driving ($\sigma_x < 0$) → $v > 0$ (tyre surface moves forward relative to road)

In the toy problem, $v$ is spatially uniform (same at every $\xi$) — a further simplification.

---

### The friction coefficient model

$$\mu(v) = \mu_d + (\mu_s - \mu_d)\exp\!\left(-\!\left(\frac{|v|}{v_S}\right)^{\delta_S}\right)$$

| Parameter | Symbol | Typical value | Meaning |
|-----------|--------|---------------|---------|
| Static friction | $\mu_s$ | 1.0 | Peak friction at zero sliding |
| Dynamic friction | $\mu_d$ | 0.7 | Friction at large sliding speed |
| Stribeck velocity | $v_S$ | 3–5 m/s | Speed where friction starts dropping |
| Stribeck exponent | $\delta_S$ | 0.5–1.0 | Shape of the drop-off |

At $v = 0$: $\mu = \mu_s$ (maximum friction). As $|v|$ grows: $\mu \to \mu_d$ (lower friction). This produces the **force peak** seen in real tyre data — friction peaks at small slip and drops at large slip.

Without the Stribeck model (constant $\mu$), the force would monotonically saturate with no peak. This doesn't match measured tyre behaviour.

---

### The bristle force and total tangential force

At each point $\xi$, the (nondimensional) bristle force per unit normal load is simply:
$$f(\xi, t) = \bar{k}_0\, z(\xi, t)$$

Stiffer bristles ($\bar{k}_0$ large) produce more force per unit deflection.

The total tangential force on the contact patch is:
$$F(t) = \int_0^L p(\xi)\, f(\xi, t)\, d\xi = \bar{k}_0\int_0^L p(\xi)\, z(\xi, t)\, d\xi$$

where $p(\xi) \geq 0$ is the **normal pressure distribution** — how the tyre load is distributed along the contact patch. For a parabolic distribution:
$$p(\xi) \propto \xi(L - \xi)$$
(zero at both edges, maximum in the middle). For a uniform distribution: $p(\xi) = F_z / L$.

The integral weights each bristle force by the local normal pressure — bristles under more load contribute more to the total friction force.

---

### The stationary (steady-state) solution

When slip $v$ is constant and the wheel has been rolling long enough, the bristle profile stops changing: $\partial z/\partial t = 0$. The PDE (B) reduces to an ODE in $\xi$:

$$V_r\frac{dz}{d\xi} = -\frac{|v|_\varepsilon}{\mu(v)}\bar{k}_0\, z - v$$

This is a first-order linear ODE. With the boundary condition $z(0) = 0$, the exact solution is:

$$z(\xi) = -\text{sgn}_\varepsilon(v)\frac{\mu(v)}{\bar{k}_0}\left[1 - \exp\!\left(-\frac{|v|_\varepsilon\,\bar{k}_0}{V_r\,\mu(v)}\xi\right)\right], \quad \xi \in [0, L]$$

where $\text{sgn}_\varepsilon(v) = v/|v|_\varepsilon$ is the regularised sign function.

**Understanding this formula:**

Define the **characteristic length**:
$$\ell_c = \frac{V_r\,\mu(v)}{|v|_\varepsilon\,\bar{k}_0}$$

Then:
$$z(\xi) = -\text{sgn}_\varepsilon(v)\frac{\mu(v)}{\bar{k}_0}\left[1 - e^{-\xi/\ell_c}\right]$$

- At the leading edge ($\xi = 0$): $z = 0$ — bristle undeformed ✓
- For $\xi \ll \ell_c$ (small $\xi$): $z \approx -\text{sgn}(v)\frac{|v|_\varepsilon}{V_r\bar{k}_0}\xi$ — linear growth (bristle bending proportional to distance from entry)
- For $\xi \gg \ell_c$ (large $\xi$, long patch or high slip): $z \to -\text{sgn}_\varepsilon(v)\frac{\mu(v)}{\bar{k}_0}$ — saturates at maximum deflection

**Physical picture:**
- At the leading edge, bristle enters straight.
- As it travels through the contact, slip bends it progressively.
- Friction simultaneously resists this bending.
- Eventually the bristle reaches a steady deflection where slip-driving and friction-restoring balance.
- The characteristic length $\ell_c = V_r\mu/|v|\bar{k}_0$ tells you how quickly this saturation happens:
  - Large $\ell_c$ (small slip, high friction): long contact needed before saturation — "adhesion zone" is large
  - Small $\ell_c$ (large slip): saturation happens near the leading edge — most of the patch is sliding

This is the FrBD equivalent of the classic "brush model" adhesion/sliding zone structure — but without an explicit partition into zones. The exponential profile describes the continuous transition naturally.

---

### Steady-state force: connection to the force-slip curve

Plugging the stationary solution into the force integral with uniform pressure $p = F_z/(2a) = F_z/L$:

$$F = \bar{k}_0\int_0^L p\,z(\xi)\,d\xi = -\text{sgn}(v)\,\mu(v)\,F_z\left[1 - \frac{\ell_c}{L}\left(1 - e^{-L/\ell_c}\right)\right]$$

This is the **force-slip curve** (steady-state friction force vs slip). Key features:
- Sign: force opposes slip direction (braking force when braking)
- At small slip ($|v| \to 0$, $\ell_c \to \infty$): $F \approx -\bar{k}_0 F_z\,L\,v/(2V_r)$ — linear in slip (no saturation yet)
- At large slip ($|v| \to \infty$, $\ell_c \to 0$): $F \to -\text{sgn}(v)\,\mu(v)\,F_z$ — saturates at $\mu(v) \times$ normal load

Combined with the Stribeck model, this gives:
- **Peak** at small-to-moderate slip (where $\mu$ is still high and patch isn't fully sliding)
- **Drop** at large slip (where $\mu \to \mu_d$)

This matches the characteristic shape of measured tyre force-slip curves and Pacejka's Magic Formula.

---

### Connection to the full 2D FrBD paper

The toy problem is Model 1 from the paper (Section 11), specialized to:
- 1D (only $x$ direction, no lateral $y$)
- Scalar states ($z, f \in \mathbb{R}$, not $\mathbb{R}^2$)
- No spin ($\varphi = 0$)
- KV bristle ($n = 0$, no internal Maxwell branches)
- Simplified transport: $\bar{V}_x = -1$ (constant), i.e., $V_r\partial/\partial x$ reduces to advection

The full paper generalises every one of these simplifications:

| Toy problem | Full paper |
|-------------|------------|
| 1D scalar ($z \in \mathbb{R}$) | 2D vector ($\boldsymbol{z} \in \mathbb{R}^2$) |
| Elastic bristle only ($f = \bar{k}_0 z$) | Viscoelastic (GM/GKV), $n$ internal states |
| No spin | Spin $\varphi$ included |
| Uniform relative velocity $v(t)$ | Spatially varying $\bar{\boldsymbol{v}}(\boldsymbol{x}, s)$ |
| 1D contact patch | 2D contact area $\mathcal{C}(t) \subset \mathbb{R}^2$ |
| $p(\xi)$ scalar weight | $p(\boldsymbol{x},s)$ 2D pressure field |

The PDE structure is identical: transport + relaxation + slip driving. The toy problem is the minimal case that preserves all essential physics while being tractable enough to understand and implement.

---

### Summary of toy problem equations

$$\boxed{
\frac{\partial z}{\partial t} + V_r\frac{\partial z}{\partial \xi} = -\frac{|v|_\varepsilon}{\mu(v)}\bar{k}_0\, z - v
}$$

| Quantity | Symbol | Role |
|---------|--------|------|
| Bristle deflection | $z(\xi, t)$ | State variable |
| Rolling speed | $V_r(t) = \omega R_r$ | Transport speed |
| Relative (slip) velocity | $v(t) = -V_r\sigma_x$ | Driving input |
| Bristle stiffness | $\bar{k}_0 > 0$ | Parameter to identify |
| Friction coefficient | $\mu(v)$ | Stribeck model (4 params) |
| Regularisation | $\varepsilon > 0$ | Numerical smoothing |
| Domain | $\xi \in [0, L]$, $t \in [0, T]$ | Contact patch × time |
| BC | $z(0, t) = 0$ | Leading edge undeformed |
| Force | $F(t) = \int_0^L p(\xi)\bar{k}_0 z(\xi,t)\,d\xi$ | Output |

Parameters to identify from data: $\bar{k}_0$, $\mu_s$, $\mu_d$, $v_S$, $\delta_S$ — total 5 scalars.

---

*Explanation written for: FrBD 2D friction paper (Romano, Nonlinear Dyn 2026). Covers full paper from first principles.*
