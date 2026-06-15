---
title: "Codebase & Algorithm Guide — from 0 to 1"
tags: [MWI, guide, onboarding, MoM, Lippmann-Schwinger, python, numpy, overview]
status: living document
date: 2026-06-15
related: "[[F1_Tutorial_2D-MoM-and-Mie-validation]], [[F2_Tutorial_CG-FFT-matrix-free-solver]], [[F1_milestone]], [[F2_milestone]]"
---

# Codebase & Algorithm Guide — from 0 to 1

> **Who this is for:** you, six months from now, opening this repo and thinking
> "what *is* all this?" It rebuilds the electromagnetic scattering algorithm from
> scratch, then walks every Python file in `mwisim/`, `scripts/`, and `tests/`,
> line-ideas explained, with **MATLAB analogies** throughout (you know MATLAB; NumPy
> is MATLAB with different punctuation). Read Part 1 to refresh the physics, Part 2 for
> the Python you need, then Part 3 is the file-by-file map.

## Contents

1. The 30-second mental model
2. The algorithm from zero (the physics)
3. NumPy for a MATLAB user (just what this repo uses)
4. File-by-file walkthrough (`mwisim/`)
5. The drivers (`scripts/`)
6. The tests as a specification (`tests/`)
7. One full run, traced end to end
8. Symbol & glossary table

---

## 1. The 30-second mental model

We simulate one experiment: **a plane microwave hits a dielectric cylinder; we compute
the scattered field on a ring of receivers around it.** Then we check our answer against
the textbook closed-form solution (the *Mie series*). If they agree, our numerical engine
is trustworthy — and *that* engine is what later reconstructs unknown objects (breast /
bone) from their scattered fields.

```
                        ┌─────────────── mwisim/ (the physics library) ───────────────┐
 parameters ──► grid ──► contrast χ ──► MoM matrix D ──► solve (I−D)E = E_inc ──► E_total
   (f, εr,        (cells)   (which cells     (cell-to-cell      (the field everywhere       │
    radius)                  are object)      coupling)          inside the object)         ▼
                                                                              scattered field at receivers
                                                                                            │
                              Mie series (analytic truth) ──────── compare ◄────────────────┘
                                                                       │
                                                                  rel L2 error  ──►  figures / pass-fail
```

- **F1** built this whole chain *densely* (a real $N\times N$ matrix). See [[F1_milestone]].
- **F2** replaced the slow matrix step with an FFT so it scales to huge grids, without
  changing any physics. See [[F2_milestone]].

---

## 2. The algorithm from zero (the physics)

### 2.1 What field are we even solving for?

In 2D **TM polarization**, the electric field points purely along $z$, so the whole
vector problem collapses to **one scalar function** $E_z(x,y)$. That is the single reason
2D-TM is the standard starting point — no vector bookkeeping, just one complex number per
point. Everything in this repo is that one scalar field, sampled on a grid.

We use the engineering time convention $e^{+j\omega t}$ (same as radar). Consequence:
outgoing waves look like $e^{-jk R}$, and the right special function is the **Hankel
function of the second kind** $H^{(2)}$. (If you ever see $H^{(1)}$ and $+i$, that's the
physics convention — don't mix them or the imaginary part flips sign.)

### 2.2 Incident, scattered, total

The field splits into two parts:

$$E_z = \underbrace{E_z^{\text{inc}}}_{\text{the wave you sent in}} + \underbrace{E_z^{\text{sc}}}_{\text{what the object re-radiates}}$$

The incident wave is a plane wave travelling along $+x$: $E_z^{\text{inc}}=E_0 e^{-jk_b x}$.
The scattered part is what we want to predict (and later, measure and invert).

### 2.3 The contrast — "how different is each point from the background?"

Define the **contrast function**

$$\chi(\mathbf r)=\frac{\varepsilon_r(\mathbf r)}{\varepsilon_b}-1.$$

It is $0$ in empty background and nonzero only inside the object. $\chi$ is the *unknown*
in the imaging problem; in the *forward* problem we know it and compute the field.

### 2.4 The master equation (Lippmann–Schwinger)

Every scattering simulation in this repo is one equation:

$$\boxed{\,E_z(\mathbf r)=E_z^{\text{inc}}(\mathbf r)+k_b^2\!\int_S G(\mathbf r,\mathbf r')\,\chi(\mathbf r')\,E_z(\mathbf r')\,dS'\,}$$

In words: **the field at a point = the wave you sent in + the sum of tiny re-radiations
from every bit of object, each weighted by how strong it is ($\chi$), how strong the field
there is ($E_z$), and how a wave travels from there to here (the Green's function $G$).**

The 2D free-space Green's function (our $e^{+j\omega t}$ convention) is

$$G(\mathbf r,\mathbf r')=\frac{1}{4j}H_0^{(2)}\!\bigl(k_b|\mathbf r-\mathbf r'|\bigr).$$

> **Why is this called nonlinear "in general" but linear here?** $E_z$ appears on *both*
> sides. In the forward problem $\chi$ is known, so it's a linear equation in the unknown
> $E_z$ — one solve and done. In the *inverse* problem $\chi$ and $E_z$ are *both* unknown
> and multiply each other → genuinely nonlinear → that's why inversion (I1–I4) needs
> iteration (Born, DBIM…).

### 2.5 Turning the integral into a matrix (Method of Moments, Richmond)

A computer can't handle a continuous integral, so we **discretize**:

1. Cover the object region with $N$ tiny **square cells** of side $d$. Assume $E_z$ and
   $\chi$ are constant inside each cell ("pulse basis").
2. Demand the equation hold exactly at each **cell center** ("point matching").
3. The integral becomes a finite sum. Collect it into matrix form:

$$(\mathbf I-\mathbf D)\,\mathbf E=\mathbf E^{\text{inc}},\qquad D_{mn}=k_b^2\,\chi_n \!\int_{\text{cell}_n}\! G(\mathbf r_m,\mathbf r')\,dS'.$$

$\mathbf E$ is the stacked field values, one per cell. $D_{mn}$ says **how much the field
in cell $n$ contributes to the field in cell $m$.** Solve the linear system → you have the
field everywhere inside the object. That is the entire forward solver.

### 2.6 The one hard integral: the self-cell

When $m=n$ (a cell's effect on itself), $\mathbf r_m=\mathbf r_n$ and $G$ blows up
($H_0^{(2)}$ is singular at zero distance). **Richmond's trick:** replace the square cell
by an equal-area disk of radius $a=d/\sqrt{\pi}$ (so $\pi a^2 = d^2$). The disk integral
has a closed form:

$$D_{mn}=\begin{cases}-\,\chi_n\,\dfrac{j\pi k_b a}{2}\,J_1(k_b a)\,H_0^{(2)}(k_b\rho_{mn}), & m\neq n,\\[2mm]-\,\chi_n\Bigl[\dfrac{j\pi k_b a}{2}\,H_1^{(2)}(k_b a)+1\Bigr], & m=n.\end{cases}$$

> ⚠️ **The famous "+1" (the F1 bug).** That extra $+1$ inside the self term came from the
> lower limit of the radial integral ($\lim_{u\to0}uH_1^{(2)}(u)=2j/\pi\neq0$). Dropping it
> matched Mie at *weak* contrast but diverged (~80% error) at strong contrast. It cost a
> day to find. In the code you'll see it as `... - 1` written *before* multiplying the
> column by $\chi_n$, so it becomes the $-\chi_n$ above. Never delete it.

### 2.7 Getting the scattered field at the receivers

Once we know $\mathbf E$ inside the object, the field at any *exterior* receiver
$\mathbf r_r$ is a plain weighted sum (no singularity, receivers are far from cells):

$$E_z^{\text{sc}}(\mathbf r_r)=k_b^2\sum_n G(\mathbf r_r,\mathbf r_n)\,\chi_n\,E_n\,dS.$$

### 2.8 The ground truth: the Mie series

For a *circular* cylinder there is an exact analytic answer — an infinite series of
angular modes:

$$E_z^{\text{sc}}(\rho,\phi)=\sum_{n=-\infty}^{\infty}(-j)^n a_n H_n^{(2)}(k_b\rho)\,e^{jn\phi},$$

with mode coefficients $a_n$ built from Bessel/Hankel functions and the boundary
conditions. We truncate the sum (it converges fast) and treat it as **truth**. The entire
validation philosophy of the project: *compare numerics to an exact answer, in simulation,
no lab needed.*

### 2.9 The validation metric

Relative $L_2$ error between our field and Mie's:

$$\text{err}=\frac{\lVert E^{\text{MoM}}-E^{\text{Mie}}\rVert_2}{\lVert E^{\text{Mie}}\rVert_2}.$$

F1 passes when this is small **and** shrinks as the grid is refined (a *convergence
curve*).

That's the whole algorithm. Everything below is this maths, written in Python.

---

## 3. NumPy for a MATLAB user (just what this repo uses)

You know MATLAB, so here is only the delta.

| Idea | MATLAB | NumPy (this repo) |
|---|---|---|
| import the library | (built in) | `import numpy as np` |
| make a vector | `linspace(0,1,5)` | `np.linspace(0,1,5)` |
| imaginary unit | `1i` or `1j` | `1j` (e.g. `4j` is one literal!) |
| matrix multiply | `A*B` | `A @ B` (the `@` operator) |
| **elementwise** multiply | `A.*B` | `A * B` |
| transpose / conj-transpose | `A.'` / `A'` | `A.T` / `A.conj().T` |
| solve $Ax=b$ | `A\b` | `np.linalg.solve(A, b)` |
| 2-norm | `norm(v)` | `np.linalg.norm(v)` |
| build a grid | `meshgrid` | `np.meshgrid(..., indexing="ij" or "xy")` |
| indexing base | **1-based**, `v(1)` | **0-based**, `v[0]` |
| index range | `v(2:5)` (incl.) | `v[1:5]` (**end-exclusive**) |
| last element | `v(end)` | `v[-1]` |
| reshape to column list | `reshape` | `arr.ravel()` / `arr.reshape(...)` |
| stack columns | `[a b]` | `np.column_stack([a, b])` |

Five gotchas that matter here:

- **`@` vs `*`.** `@` is real matrix multiply (linear algebra); `*` is elementwise
  (`.*`). Mixing them up is the #1 NumPy bug for MATLAB users.
- **Broadcasting.** `centers[:, None, :] - centers[None, :, :]` is how we build an
  all-pairs difference with no loop. `None` inserts a new axis (like making a
  "page" dimension), and NumPy auto-stretches singleton axes. This single line is what
  vectorizes the distance matrix.
- **`indexing="ij"` vs `"xy"`.** `"ij"` = matrix/row-column order (axis 0 is $y$ rows);
  `"xy"` = Cartesian (axis 0 is $x$). The repo deliberately uses each in the right place;
  the FFT operator depends on the ravel order matching.
- **Slicing is end-exclusive and 0-based.** `vp[:ny, :nx]` keeps rows `0..ny-1`. This is
  how F2 crops the FFT result.
- **`dtype=complex`.** Arrays must be told they're complex up front, or assigning a
  complex value silently drops the imaginary part.

A few Python-isms you'll see:

- `def f(x: float) -> np.ndarray:` — the `: float` / `-> np.ndarray` are **type hints**,
  documentation only; Python doesn't enforce them.
- `"""..."""` right under a `def` is the **docstring** (help text).
- `from __future__ import annotations` — a harmless compatibility line; ignore it.
- `a or b` — returns `a` if it's "truthy", else `b`. Used for defaults: `P["R_cyl"] or 0.5*lam0`.

---

## 4. File-by-file walkthrough (`mwisim/`)

> Mapping: §2.x physics → file. grid/contrast → §2.3, 2.5 · green → §2.4 · mom → §2.5–2.7
> · mie → §2.8 · metrics → §2.9 · operators → F2 (§2.5 sped up).

### 4.1 `grid.py` — lay down the cells, mark the object

**`make_grid(domain_size, d)`** builds the $N$ cell centers covering a square
$[-L/2,L/2]^2$ and returns them as an `(N, 2)` array of $(x,y)$ plus the cell area
`dS = d**2`.

- `N_cells = math.ceil(domain_size / d)` — how many cells per side.
- The `if N_cells % 2 == 0` branch just centers the grid nicely on the origin whether the
  count is even or odd (`%` is modulo; `//` is integer division).
- `X, Y = np.meshgrid(x_, y_, indexing="xy")` then
  `centers = np.column_stack([X.ravel(), Y.ravel()])` — flatten the 2D grid into a flat
  list of points. *MATLAB:* `[X(:) Y(:)]`.

**`assign_contrast(centers, R_cyl, eps_r, eps_b=1.0)`** decides which cells are "object."

- `r = np.hypot(centers[:,0], centers[:,1])` — distance of each cell to the origin
  (`hypot` = $\sqrt{x^2+y^2}$, the whole vector at once).
- `chi = np.where(r <= R_cyl, eps_r/eps_b - 1, 0.0)` — *vectorized if/else*: where the
  cell is inside the radius use $\varepsilon_r/\varepsilon_b-1$, else $0$. This is §2.3.
- `.astype(complex)` — force complex storage so lossy $\varepsilon_r$ works.

### 4.2 `green.py` — how a wave travels from one point to another

**`green_2d(k_b, R)`** returns $G(R)=\frac{1}{4j}H_0^{(2)}(k_b R)$ — §2.4, exactly.
`hankel2(0, k_b*R)` is SciPy's $H_0^{(2)}$. Note `1/4j` parses as `1/(4j)` (since `4j` is
a single complex literal), which is correct. The self term $R=0$ is *not* handled here —
it's done specially in `mom.py` (§2.6).

### 4.3 `mom.py` — the heart: build the matrix, solve, radiate

This file is the forward solver (§2.5–2.7). Four functions:

**`build_D(centers, chi, k_b, d)`** → the $N\times N$ coupling matrix (§2.5–2.6). The file
keeps a commented-out *slow, readable* double loop on top (great for understanding) and a
*fast vectorized* version below. The fast one:

```python
a = d / np.sqrt(np.pi)                       # equal-area disk radius (§2.6)
pref = -(1j * np.pi * k_b * a / 2)           # the shared prefactor in the boxed formula
diff = centers[:, None, :] - centers[None, :, :]   # all-pairs (r_m - r_n), shape (N,N,2)
rho  = np.sqrt((diff**2).sum(axis=-1))       # all-pairs distances rho_mn, shape (N,N)
D = pref * jv(1, k_b*a) * hankel2(0, k_b*rho)      # off-diagonal entries
np.fill_diagonal(D, pref * hankel2(1, k_b*a) - 1)  # SELF term incl. the famous "-1" (§2.6!)
D = D * chi[None, :]                          # multiply each COLUMN n by chi_n
return D
```

Why column multiply (`chi[None, :]`)? Because $\chi_n$ belongs to the *source* cell $n$,
which is the **column** index. `chi[None, :]` is a `(1, N)` row that broadcasts down every
row. After this, the diagonal `-1` has become $-\chi_n$ — that is the self-cell term.

**`incident_plane_wave(centers, k_b, E0=1.0)`** → $E_0 e^{-jk_b x}$ at each cell
(§2.2). `centers[:,0]` is the $x$ column. One line, fully vectorized.

**`solve_total_field(D, E_inc)`** → solves $(\mathbf I-\mathbf D)\mathbf E=\mathbf E^{\text{inc}}$
with `np.linalg.solve(I - D, E_inc)` (*MATLAB:* `(I-D)\E_inc`). This is the dense direct
solve that F2 later replaces. Returns the field inside the object.

**`scattered_field(rx_points, centers, chi, E_tot, k_b, dS)`** → the field at exterior
receivers (§2.7): builds receiver-to-cell distances by broadcasting, forms $G$, then
`E_sc = pref * (G @ (chi * E_tot))`. Here `G @ (...)` is a real matrix-vector product (the
weighted sum), while `chi * E_tot` is elementwise. Receivers are outside the object so no
singularity.

### 4.4 `mie.py` — the analytic ground truth (§2.8)

**`mie_an(n, k_b, k_1, R_cyl)`** computes one mode coefficient $a_n$ from the
boundary-condition ratio of Bessel/Hankel terms. `jv, jvp` are $J_n$ and its derivative;
`hankel2, h2vp` are $H_n^{(2)}$ and its derivative.

**`mie_scattered(rx_points, k_b, k_1, R_cyl, Nmax=None)`** sums the series over modes
$-N_{\max}..N_{\max}$:

- `rho, phi = np.hypot(...), np.arctan2(...)` — convert receivers to polar.
- default `Nmax = ceil(|k_b| R_cyl) + 10` — enough modes for convergence (more modes for
  bigger/faster-varying objects).
- the `for n in range(-Nmax, Nmax+1)` loop adds each mode's contribution.
- *sanity check baked into the tests:* increase `Nmax` and the answer must stop changing
  (T5).

> **Get this file right FIRST.** If the "truth" is wrong, every comparison downstream
> lies. That's why Mie has its own self-convergence test before MoM is ever trusted.

### 4.5 `metrics.py` — score and convergence (§2.9)

**`rel_l2_error(approx, ref)`** = $\lVert \text{approx}-\text{ref}\rVert / \lVert \text{ref}\rVert$,
complex-aware. A helper, used everywhere.

**`convergence_study(d_list, params)`** runs the *entire* pipeline for several cell sizes
$d$ and records the error each time, returning `(cells_per_wavelength, errors)` for the
log-log convergence plot. Note the line `m = chi != 0; centers, chi = centers[m], chi[m]`
— **boolean-mask indexing** keeps only object cells to shrink the dense system (MATLAB
logical indexing `centers(m,:)`). The derived wavenumbers use
$k=2\pi f/c\cdot\sqrt{\varepsilon}$.

### 4.6 `operators.py` — the F2 fast engine (same physics, FFT speed)

This is the F2 deliverable; full theory in [[F2_Tutorial_CG-FFT-matrix-free-solver]] and
[[F2_milestone]]. The key realisation: on a regular grid $D_{mn}=g(\mathbf r_m-\mathbf r_n)\chi_n$
depends only on the **displacement** $\mathbf r_m-\mathbf r_n$, so $\mathbf D$ is
**block-Toeplitz with Toeplitz blocks (BTTB)**, and "multiply by a Toeplitz matrix" = "do a
convolution" = "multiply in Fourier space." So we never store $\mathbf D$:

- **`infer_grid_shape(centers)`** — recover $(N_y,N_x)$ from the flat `centers` (the FFT
  needs the regular grid back in 2D).
- **`GreenFFT.__init__`** — compute the kernel $g$ over *all* displacements once,
  zero-pad it to `next_fast_len(2N-1)` (the **circulant embedding** that prevents FFT
  wrap-around), and cache `self.G_hat = fft2(g_pad)`. The self entry carries the same
  `- 1` as `build_D`.
- **`_conv(v_grid)`** — one convolution: pad `v` into a zeros canvas, `ifft2(G_hat *
  fft2(vp))`, then crop `[:ny, :nx]`. The wrap-around garbage lives entirely in the
  discarded part.
- **`apply_D(x)`** — reshape `x` to a grid, multiply by $\chi$ (source/column), convolve,
  ravel back. This *equals* `build_D @ x` but in $O(N\log N)$ (test T9 proves it).
- **`apply_IminusD(x)`** — `x - apply_D(x)`: the operator $(\mathbf I-\mathbf D)$.
- **`as_linear_operator()`** — wraps the above so SciPy's iterative solvers can call it
  with just matvecs (no matrix needed).
- **`solve_total_field(...)`** — BiCGStab or GMRES iterates to the answer; a callback
  counts iterations; it reports an honest residual $\lVert b-A\mathbf E\rVert/\lVert b\rVert$.

Also in this file: **`A_op` / `AH_op`** — the Born forward operator and its adjoint for the
*inversion* stage (I1+). `A_op(v) = k_b^2 dS · G_tr @ (E_inc * v)` maps a contrast guess to
predicted receiver fields; `AH_op` back-projects a residual onto the grid. These are the
seeds of the imaging work to come.

### 4.7 `inverse/__init__.py` — placeholder for I1–I4

Empty package with a docstring roadmap (Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM). Nothing
to run yet; this is where I1 will land.

---

## 5. The drivers (`scripts/`)

These are *orchestration only* — they import the library and make figures. No physics
lives here.

### 5.1 `run_f1.py`

- A parameter dict `P` (frequency, $\varepsilon_r$, geometry). `_derived(P)` turns it into
  wavelengths and wavenumbers.
- `run_pointwise(P)` runs one grid, computes MoM and Mie scattered fields on the receiver
  ring, prints the error, and saves `docs/fig_pointwise.png` (Re/Im overlay: dots = MoM,
  line = Mie).
- `run_convergence(P)` calls `convergence_study` over several cell sizes and saves
  `docs/fig_convergence.png` (log-log error vs cells-per-wavelength — should slope down).
- `sys.path.insert(0, ...)` at the top is the "no `pip install` needed" trick: it adds the
  repo root to Python's import search path so `import mwisim...` works.

### 5.2 `run_f2.py`

The F2 benchmark. `setup`/`bench` build a problem at a target grid size, time both the
**CG-FFT** path and (when it still fits in RAM) the **dense** path, record iterations and
memory, and confirm they agree (`match` column). `main()` sweeps a list of sizes, prints
the table you saw, and saves `docs/fig_f2_scaling.png` (time and memory vs $N$, dense vs
FFT). The headline line at the end reports the largest case.

---

## 6. The tests as a specification (`tests/`)

The tests *are* the contract — read them as "what correct looks like." They're written for
`pytest`; run `python -m pytest -q` (the `python -m` form makes `import mwisim` resolve).

### 6.1 `test_f1.py` (T1–T8) — does the forward solver match physics?

- **helper / T2** — incident wave has unit magnitude.
- **T3** — Green depends only on distance (equal distances → equal values).
- **T4** — *weak* scatterer: total field ≈ incident field (object barely perturbs).
- **T5** — Mie self-convergence: `Nmax=8` vs `Nmax=25` agree (truth is trustworthy).
- **T6/T7** — MoM matches Mie for a weak cylinder (<5%).
- **T8** — MoM *still* matches Mie for a **strong** cylinder ($\varepsilon_r=8$) — this is
  the test the self-cell "+1" bug used to fail.

### 6.2 `test_f2.py` (T9–T14) — does the fast path equal the slow path?

- **T9** — `apply_D @ x` equals `build_D @ x` on a *random* complex vector (<1e-12 target;
  your run ~1e-8). Random input is deliberate: it stresses every matrix entry.
- **T10** — $(\mathbf I-\mathbf D)\mathbf x$ matches.
- **T11/T12** — BiCGStab and GMRES solves match the dense direct solve.
- **T13** — end-to-end: fast-path scattered field = slow-path = Mie.
- **T14** — `infer_grid_shape` works and rejects a non-rectangular grid.

> Note F1 tests **mask** to object cells (smaller dense system); F2 tests use the **full
> regular grid** because the FFT structure needs it. The $\chi=0$ exterior cells cost
> almost nothing.

---

## 7. One full run, traced end to end

Following `run_f1.run_pointwise` with the default weak cylinder:

1. `_derived(P)` → $\lambda_0$, $k_b$, $k_1$, cylinder radius, receiver radius.
2. `make_grid` → ~thousands of cell centers; `assign_contrast` → $\chi$ (nonzero inside).
3. mask to object cells → smaller system.
4. `build_D` → the $N\times N$ coupling matrix (§2.5–2.6).
5. `incident_plane_wave` → $\mathbf E^{\text{inc}}$.
6. `solve_total_field` → solve $(\mathbf I-\mathbf D)\mathbf E=\mathbf E^{\text{inc}}$ → field inside (§2.5).
7. `scattered_field` at the receiver ring → predicted $E^{\text{sc}}$ (§2.7).
8. `mie_scattered` at the same ring → analytic truth (§2.8).
9. `rel_l2_error` → the score; plot dots vs line.

F2 only changes steps 4+6: instead of building `D` and factorising, `GreenFFT` applies
$\mathbf D$ via FFT and iterates. Steps 1–3, 5, 7–9 are untouched.

---

## 8. Symbol & glossary table

| Symbol / name | Meaning |
|---|---|
| $E_z$ | the scalar field we solve for (2D TM) |
| $E^{\text{inc}}, E^{\text{sc}}$ | incident (sent-in) and scattered (re-radiated) field |
| $\chi$ (`chi`) | contrast $\varepsilon_r/\varepsilon_b-1$; nonzero inside object |
| $k_b, k_1$ | wavenumber in background, inside cylinder |
| $G$ | 2D Green's function $\frac{1}{4j}H_0^{(2)}(k_bR)$ — point-to-point wave travel |
| $\mathbf D$ | MoM coupling matrix; $(\mathbf I-\mathbf D)\mathbf E=\mathbf E^{\text{inc}}$ |
| $d$, $a$, `dS` | cell side, equal-area disk radius $d/\sqrt\pi$, cell area $d^2$ |
| $\rho_{mn}$ | distance between cell $m$ and cell $n$ |
| Mie series | exact analytic scattered field for a circular cylinder (ground truth) |
| BTTB | block-Toeplitz w/ Toeplitz blocks — structure that makes $\mathbf D$ an FFT |
| MoM | Method of Moments (Richmond) — the discretization scheme |
| L-S | Lippmann–Schwinger — the master integral equation |
| BiCGStab / GMRES | iterative linear solvers that need only matrix-vector products |
| $H_n^{(2)}, J_n, Y_n$ | Hankel (2nd kind), Bessel 1st kind, Bessel 2nd kind |

---

*Living document — update as F3/inversion/HLS land. See [[F1_Tutorial_2D-MoM-and-Mie-validation]]
and [[F2_Tutorial_CG-FFT-matrix-free-solver]] for the full derivations behind §2 and §4.6.*
