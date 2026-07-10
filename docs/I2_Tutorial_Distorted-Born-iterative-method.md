---
title: "I2 Tutorial: DBIM — the nonlinear χ-map (Distorted Born Iterative Method)"
tags: [MWI, MWT, inversion, DBIM, BIM, nonlinear-inverse, Frechet, distorted-Green, LSMR, Tikhonov, tutorial]
status: tutorial v1
date: 2026-07-05
related: "[[PROJECT_PLAN]], [[I1_Tutorial_Born-linear-inversion]], [[I1_milestone]], [[CODE_GUIDE_codebase-and-algorithm-from-zero]], [[F1_Tutorial_2D-MoM-and-Mie-validation]], [[F2_Tutorial_CG-FFT-matrix-free-solver]]"
---
# I2 Tutorial: DBIM — the nonlinear $\chi$-map

> **How to use this tutorial:** read it once for the whole arc, then implement the TODO-marked pieces in `mwisim/inverse/dbim.py` against the given scaffolding, returning here when stuck. It gives the physics, the linear-algebra of the update, the algorithm, a function checklist, and self-tests — **not** a full solution. When the I2 tests go green you get the project's first *quantitative* image of a scatterer that is **too strong for Born** — the headline result.

> **Where I2 sits.** I1 solved the inverse problem *once*, under the Born (weak-scatterer) approximation. Real tissue is not weak, so a single Born step has large **model error** (I1 milestone §3). DBIM removes that error by **iterating**: guess $\chi$, run the *full* forward solver to see how wrong the data is, linearize **around the current guess** (not around empty space), take a regularized least-squares step, repeat. Every inner step is I1's machinery — DBIM is "I1 in a loop, re-linearized each time." Build the loop on top of the engine you already have.

---

## 0. Goal and acceptance criteria

**Goal:** from multi-view scattered-field measurements of a scatterer strong enough that Born fails (ε_r ≈ 1.5–2), reconstruct $\hat{\chi}$ by the **Distorted Born Iterative Method** — an outer loop that, at each step, (a) solves the *full* nonlinear forward problem for the current $\chi$, (b) forms the data residual, (c) builds the **distorted Born operator** (the Fréchet derivative of the forward map at the current $\chi$), and (d) solves one regularized linear least-squares problem (LSMR, exactly as in I1) for the update $\Delta \chi$.

**Acceptance criteria (I2 passes when all hold):**

1. **Forward-sim consistency:** `simulate_scattered_data(χ_true)` reproduces the problem's measured data `d` (both are the same full MoM forward path) to ~1e-10 — proves your forward re-simulation is wired correctly.
2. **Distorted-Born adjoint test:** the Fréchet operator `J` and its adjoint satisfy $\left\langle J v, u\right\rangle=\left\langle v, J^H u\right\rangle$ to ~1e-10 on random vectors (without this the inner LSMR cannot converge — same gate as I1.2).
3. **DBIM beats Born (the headline):** on a moderate scatterer where single-step Born is poor, DBIM's reconstruction explains the *true nonlinear* data far better — full-forward data residual $\lVert\mathbf d-\mathcal F(\hat\chi)\rVert/\lVert\mathbf d\rVert$ for DBIM is a small fraction of Born's, and the $\chi$-error is lower.
4. **Convergence:** the outer data-residual history is (overall) decreasing and reaches a small value within a modest number of outer iterations.
5. `build("inverter","dbim")` produces a working inverter.

---

## 1. Why one Born step is not enough — the picture

I1's linear model was $\mathbf d\approx\mathbf A\chi$, where $\mathbf A$ was built by replacing the true interior field $E$ with the *incident* field $E^{\text{inc}}$ (valid only when the object barely perturbs the wave). For a real scatterer the interior field is strongly reshaped by the object (multiple scattering), so $E\ne E^{\text{inc}}$ and the linear model is simply **wrong** — no amount of clever least-squares fixes a wrong model (CODE_GUIDE, §"why regularized LS and what beats it", axis 2).

The fix is not a better estimator but a better *model*, applied repeatedly. Think of Newton's method for a nonlinear equation $f(x)=0$: you can't solve it in one shot, so you linearize at the current $x$, step, re-linearize at the new $x$, and repeat. DBIM is exactly Newton/Gauss–Newton for the scattering inverse problem: the "function" is the forward map $\chi\mapsto\mathbf d$, and each outer iteration re-linearizes it around the latest $\chi$.

---

## 2. The exact forward map (what "the truth" costs)

Everything in §3–§5 is built from **three operators** and one nonlinear map. Fix these first — the rest is bookkeeping.

> **Notation (memorize these four things).**
> - $\text{diag}(\mathbf a)$ = the diagonal matrix with vector $\mathbf a$ on the diagonal; $\text{diag}(\mathbf a)\,\mathbf b=\mathbf a\odot\mathbf b$ = **elementwise** product (multiply component by component). This is just "scale each cell by its own number."
> - $\mathcal G$ (the **grid-to-grid** Green operator, $N\times N$): "a unit current sitting in each cell radiates to every other cell through the *empty background*." It's `build_D` with $\chi\equiv1$ (the pure geometry, no contrast). Symmetric, because cell-to-cell travel depends only on distance.
> - $\mathcal S$ (the **grid-to-receiver** operator, $M\times N$): "a current in each cell radiates out to the $M$ receivers through the empty background." $\mathcal S=k_b^2\,dS\,G_{\text{tr}}$, the thing inside `scattered_field` / `A_op`.
> - $\mathbf D(\chi)=\mathcal G\,\text{diag}(\chi)$ (the MoM domain operator, `build_D`): "first weight each cell by its contrast $\chi$, then let those weighted currents radiate to all cells." This is what couples the cells to each other.

With those, the forward map $\mathcal F:\chi\mapsto\mathbf d$ is **two steps per incidence $i$**:

$$\underbrace{(\mathbf I-\mathbf D(\chi))\,\mathbf E_i=\mathbf E^{\text{inc}}_i}_{\text{(A) solve the interior total field}},\qquad \underbrace{\mathbf d_i=\mathcal S\,\big(\chi\odot\mathbf E_i\big)=\mathcal S\,\text{diag}(\chi)\,\mathbf E_i}_{\text{(B) radiate to the receivers}}.$$

- **Step (A)** is F1/F2's forward solve: it finds the *actual* field $\mathbf E_i$ **inside** the object (the wave, reshaped by multiple scattering). \
- **Step (B)** takes the induced currents $\chi\odot\mathbf E_i$ and radiates them out to the receivers to get the measured scattered field. Stacking all incidences gives $\mathbf d=\mathcal F(\chi)$.

**The one fact that makes this nonlinear:** in step (B), $\chi$ appears *twice* — once explicitly (the $\text{diag}(\chi)$), and once **hidden inside $\mathbf E_i$**, because $\mathbf E_i$ was produced by a solve that itself contains $\chi$ (step A). I1's Born approximation pretended the hidden copy didn't exist (it froze $\mathbf E_i=\mathbf E^{\text{inc}}_i$, the object-free field). DBIM keeps both copies — that's the whole difference, and §3 is just "differentiate carefully, keeping both."

Unlike I1's cheap $\mathbf A\chi$, evaluating $\mathcal F$ costs a *full forward solve per incidence* — the price of the truth, and why the forward engine had to exist and be fast first (F1/F2).

---
## 3. The Fréchet derivative — the distorted Born operator (the heart of DBIM)

> **What "Fréchet derivative" means (read this first).** It is just the **Jacobian of the forward map**, generalized from single numbers to vector/function inputs — the best *linear* approximation of a map near a point. Same idea as ordinary calculus at every scale: scalar $f(x+\delta x)\approx f(x)+f'(x)\,\delta x$ (slope = a number $f'(x)$); vector $\mathbf F(\mathbf x+\delta\mathbf x)\approx\mathbf F(\mathbf x)+J(\mathbf x)\,\delta\mathbf x$ (slope = a Jacobian matrix $J(\mathbf x)$); function-input $\mathcal F(x+\delta x)=\mathcal F(x)+A\,\delta x+o(\lVert\delta x\rVert)$ (slope = a linear operator $A$, the Fréchet derivative). The word appears because in the *continuous* physics $\chi$ is a function; in our *discretized* code $\chi$ is a length-$N$ vector, so the Fréchet derivative $J$ is literally the $(N_v M)\times N$ **Jacobian matrix** of the forward map. Everywhere below, read "Fréchet derivative" as "Jacobian of $\mathcal F$." (The finite-difference check $J\,\delta\chi\approx[\mathcal F(\chi+\varepsilon\,\delta\chi)-\mathcal F(\chi)]/\varepsilon$ is exactly this definition, tested numerically.)

**The whole goal of §3:** nudge $\chi\to\chi+\delta\chi$ and find, to first order, the resulting nudge in the data $\delta\mathbf d_i$. The operator that maps $\delta\chi\mapsto\delta\mathbf d_i$ is $J_i$, the distorted Born operator. We build it in four small moves.

### 3.1 Move 1 — differentiate step (B), and get two terms

Data: $\mathbf d_i=\mathcal S\,\text{diag}(\chi)\,\mathbf E_i$. Both $\chi$ and $\mathbf E_i$ change when we nudge $\chi$, so use the **product rule** (exactly like $(fg)'=f'g+fg'$):

$$\delta\mathbf d_i=\mathcal S\big[\,\underbrace{\text{diag}(\delta\chi)\,\mathbf E_i}_{\text{term A: contrast changed}}+\underbrace{\text{diag}(\chi)\,\delta\mathbf E_i}_{\text{term B: field changed}}\,\big].$$
where, $\mathcal S=k_b^2\,dS\,G_{\text{tr}}$.
- **Term A** (the "direct" effect): the contrast itself moved by $\delta\chi$, with the field held fixed. This is the *only* term Born/I1 keeps.
- **Term B** (the "indirect" effect): changing the object also **reshapes the interior field** ($\delta\mathbf E_i\ne0$), and that reshaped field radiates too. This term is what Born throws away — and what makes DBIM correct for strong scatterers. We still need a formula for $\delta\mathbf E_i$; that's Move 3.

### 3.2 Move 2 — the elementwise-commute identity (answers "why $\text{diag}(\delta\chi)\mathbf E_i=\text{diag}(\mathbf E_i)\delta\chi$")

Elementwise multiplication commutes: $(\mathbf a\odot\mathbf b)_n=a_n b_n=b_n a_n=(\mathbf b\odot\mathbf a)_n$. So for **any** two vectors,

$$\text{diag}(\mathbf a)\,\mathbf b=\mathbf a\odot\mathbf b=\mathbf b\odot\mathbf a=\text{diag}(\mathbf b)\,\mathbf a.$$

> **Concrete example.** Let $\mathbf a=[2,5,10]^{\mathsf T}$ and $\mathbf b=[3,4,7]^{\mathsf T}$. The elementwise product multiplies matching entries: $\mathbf a\odot\mathbf b=[2{\cdot}3,\,5{\cdot}4,\,10{\cdot}7]^{\mathsf T}=[6,20,70]^{\mathsf T}$. Writing it as $\text{diag}(\mathbf a)\,\mathbf b$ gives the same thing, because the off-diagonal zeros kill all the cross terms of matrix multiplication:
> $$\text{diag}(\mathbf a)\,\mathbf b=\begin{bmatrix}2&0&0\\0&5&0\\0&0&10\end{bmatrix}\begin{bmatrix}3\\4\\7\end{bmatrix}=\begin{bmatrix}6\\20\\70\end{bmatrix}.$$
> Row $n$ is just $a_n\times b_n$ — nothing mixes between rows (unlike a full matrix–vector product, where every output entry blends all of $\mathbf b$). That is exactly "**scale each cell $n$ by its own number $a_n$**." And swapping the roles — putting $\mathbf b$ on the diagonal — gives $\text{diag}(\mathbf b)\,\mathbf a=[3{\cdot}2,\,4{\cdot}5,\,7{\cdot}10]^{\mathsf T}=[6,20,70]^{\mathsf T}$, the **same** vector: that's the commute. (Physical read: with $\chi$ = "how strongly each cell scatters" and $\mathbf E_i$ = "the field in each cell," $\text{diag}(\chi)\mathbf E_i=\chi\odot\mathbf E_i$ is the induced current per cell = strength × local field, computed cell by cell.)

Apply it to term A with $\mathbf a=\delta\chi$ and $\mathbf b=\mathbf E_i$: $\ \text{diag}(\delta\chi)\,\mathbf E_i=\text{diag}(\mathbf E_i)\,\delta\chi$. Nothing changed numerically — we just **moved $\delta\chi$ to the right** so it becomes the vector every term acts *on*. That is what lets us factor out a single operator times $\delta\chi$ at the end (an operator has to act on a common right-hand vector, and we want that vector to be $\delta\chi$).

### 3.3 Move 3 — differentiate step (A) to get $\delta\mathbf E_i$

The interior field obeys $(\mathbf I-\mathbf D)\,\mathbf E_i=\mathbf E^{\text{inc}}_i$ with $\mathbf D=\mathcal G\,\text{diag}(\chi)$. The incident field $\mathbf E^{\text{inc}}_i$ doesn't depend on $\chi$, so its nudge is $\mathbf 0$. Differentiate both sides (product rule again; $\delta\mathbf D=\mathcal G\,\text{diag}(\delta\chi)$):

$$-\,\delta\mathbf D\,\mathbf E_i+(\mathbf I-\mathbf D)\,\delta\mathbf E_i=\mathbf 0\ \Rightarrow\ (\mathbf I-\mathbf D)\,\delta\mathbf E_i=\mathcal G\,\text{diag}(\delta\chi)\,\mathbf E_i.$$

Solve for the field-nudge (and use Move 2 to put $\delta\chi$ on the right):

$$\delta\mathbf E_i=(\mathbf I-\mathbf D)^{-1}\mathcal G\,\text{diag}(\delta\chi)\,\mathbf E_i=(\mathbf I-\mathbf D)^{-1}\mathcal G\,\text{diag}(\mathbf E_i)\,\delta\chi.$$

Read it physically: a small contrast bump $\delta\chi$ creates a small extra current $\text{diag}(\mathbf E_i)\delta\chi$ on the grid; $\mathcal G$ radiates it to the other cells; and $(\mathbf I-\mathbf D)^{-1}$ lets that ripple **bounce around inside the object** (multiple scattering) before settling — that inverse *is* the "the object talks to itself" factor.

### 3.4 Move 4 — substitute and factor → the distorted operator (answers "where does $\mathcal S^{\text{dist}}$ come from")

Put term A (via Move 2) and term B (via Move 3) back into $\delta\mathbf d_i$:

$$\delta\mathbf d_i=\underbrace{\mathcal S\,\text{diag}(\mathbf E_i)\,\delta\chi}_{\text{term A}}+\underbrace{\mathcal S\,\text{diag}(\chi)(\mathbf I-\mathbf D)^{-1}\mathcal G\,\text{diag}(\mathbf E_i)\,\delta\chi}_{\text{term B}}.$$

Both terms end in the **same right factor** $\text{diag}(\mathbf E_i)\,\delta\chi$ and start with $\mathcal S$. Factor them out (term A carries an implicit $\mathbf I$):

$$\delta\mathbf d_i=\underbrace{\mathcal S\big[\,\mathbf I+\text{diag}(\chi)(\mathbf I-\mathbf D)^{-1}\mathcal G\,\big]}_{\displaystyle \mathcal S^{\text{dist}}}\,\text{diag}(\mathbf E_i)\,\delta\chi.$$

So $\mathcal S^{\text{dist}}$ is nothing mysterious — it is exactly "term A ($\mathbf I$) + term B (the multiple-scattering piece)" collected into one receiver operator. The $\mathbf I$ says "radiate the extra current straight out to the receivers"; the second piece says "…and *also* account for that current rattling around inside the object first, then leaking out." That is the boxed Fréchet derivative:

$$\boxed{\ \delta\mathbf d_i=\mathcal S^{\text{dist}}\big(\mathbf E_i\odot\delta\chi\big)=J_i\,\delta\chi\ },\qquad \mathcal S^{\text{dist}}=\mathcal S\big[\mathbf I+\text{diag}(\chi)(\mathbf I-\mathbf D)^{-1}\mathcal G\big].$$

### 3.5 The I1 ↔ I2 correspondence (answers "is $\mathbf A_i=J_i$? what differs?")

Same *shape* — "receiver operator × diag(field)" — but different ingredients:

$$\text{I1 (Born):}\quad \mathbf A_i=\mathcal S\,\text{diag}(\mathbf E^{\text{inc}}_i)\qquad\longrightarrow\qquad \text{I2 (DBIM):}\quad J_i=\mathcal S^{\text{dist}}\,\text{diag}(\mathbf E_i).$$

**No, $\mathbf A_i\ne J_i$ in general.** Two things differ:

| piece             | I1 / Born                             | I2 / DBIM                              | difference in one line                                                                                                                              |
| ----------------- | ------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| receiver operator | $\mathcal S$ (homogeneous)            | $\mathcal S^{\text{dist}}$ (distorted) | $\mathcal S$: grid→receiver through **empty space**; $\mathcal S^{\text{dist}}$: **through the current object** (adds the multiple-scattering term) |
| field             | $\mathbf E^{\text{inc}}_i$ (incident) | $\mathbf E_i$ (total)                  | $\mathbf E^{\text{inc}}$: field with **no object** (known, free); $\mathbf E_i$: **actual** field inside the object (needs a forward solve)         |

They **coincide exactly when $\chi=\mathbf 0$** (empty background): then $\mathbf D=\mathbf 0$, so $\mathcal S^{\text{dist}}=\mathcal S[\mathbf I+\mathbf 0]=\mathcal S$ and $\mathbf E_i=(\mathbf I-\mathbf 0)^{-1}\mathbf E^{\text{inc}}_i=\mathbf E^{\text{inc}}_i$, giving $J_i=\mathbf A_i$. That is why **DBIM's first step (from $\chi_0=\mathbf 0$) is precisely one Born step**; each later step re-computes $\mathcal S^{\text{dist}}$ and $\mathbf E_i$ at the *current* object, so $J$ drifts away from $\mathbf A$ and tracks the true nonlinear physics. Because $J_i$ has the identical *form* as $\mathbf A_i$, the code reuses I1's `BornOperator`/`A_op`/`AH_op` untouched — just feed them $G_{\text{tr}}^{\text{dist}}$ (from §4) and the total fields $\mathbf E_i$.

---

## 4. Computing the distorted Green operator by reciprocity (the given helper)

$\mathcal S^{\text{dist}}$ contains an $(\mathbf I-\mathbf D)^{-1}$ — an $N\times N$ inverse, which sounds ruinous. The escape: **we never need the whole inverse.** $\mathcal S^{\text{dist}}$ is only $M\times N$ ($M$ = number of receivers, a few dozen), so we build it **one receiver-row at a time**, and every row solve shares one factorization of $(\mathbf I-\mathbf D)$.

**Row $m$, step by step.** Split off the second term of $\mathcal S^{\text{dist}}=\mathcal S+\mathcal S\,\text{diag}(\chi)(\mathbf I-\mathbf D)^{-1}\mathcal G$ and look at just row $m$ (a $1\times N$ covector). Let $\mathbf s_m^{\mathsf T}=\mathcal S[m,:]$ be that receiver's homogeneous row. The second term's row $m$ is $\ \mathbf s_m^{\mathsf T}\,\text{diag}(\chi)\,(\mathbf I-\mathbf D)^{-1}\mathcal G$. Define

$$\mathbf z_m^{\mathsf T}=\big(\chi\odot\mathbf s_m\big)^{\mathsf T}(\mathbf I-\mathbf D)^{-1}\quad\Longleftrightarrow\quad (\mathbf I-\mathbf D)^{\mathsf T}\mathbf z_m=\chi\odot\mathbf s_m$$

(just moving the inverse to the other side turns it into a **linear solve** — no explicit inverse). Then, using that $\mathcal G$ is symmetric ($\mathbf z_m^{\mathsf T}\mathcal G=(\mathcal G\,\mathbf z_m)^{\mathsf T}$),

$$\mathcal S^{\text{dist}}[m,:]=\mathbf s_m^{\mathsf T}+(\mathcal G\,\mathbf z_m)^{\mathsf T}.$$

So **per receiver: one solve** $(\mathbf I-\mathbf D)^{\mathsf T}\mathbf z_m=\chi\odot\mathbf s_m$, then one $\mathcal G\,\mathbf z_m$. Factor $(\mathbf I-\mathbf D)$ once, back-substitute $M$ times — cheap.

**Why this is called reciprocity (the physical picture).** $\mathbf z_m$ is "the field that appears on the grid when you put a source at **receiver $m$** and let it scatter off the current object." Reciprocity (a source at A produces at B the same as a source at B produces at A) says: instead of the naive recipe "excite each of the $N$ grid cells and see what reaches receiver $m$" ($N$ solves), you may equivalently "excite from receiver $m$ and see what reaches each grid cell" (**1 solve**). Since receivers ($M$) are far fewer than cells ($N$), you do $M$ solves instead of $N$ — that is the whole saving.

**Vectorized (all receivers at once, shared factor):**

$$Z=(\mathbf I-\mathbf D)^{-\mathsf T}\big[\text{diag}(\chi)\,\mathcal S^{\mathsf T}\big],\qquad \mathcal S^{\text{dist}}=\mathcal S+(\mathcal G\,Z)^{\mathsf T}.$$

Finally, `BornOperator` re-multiplies by $k_b^2 dS$ internally, so the matrix it actually wants is $G_{\text{tr}}^{\text{dist}}=\mathcal S^{\text{dist}}/(k_b^2\,dS)$. **This routine (`distorted_green_matrix`) is GIVEN** — numerically delicate, not the learning point of I2. Study it against the four steps above; you don't have to write it. (It's finite-difference-verified to match the true Jacobian.)

> **BIM vs DBIM (the one knob).** Drop the second term — keep $\mathcal S^{\text{dist}}\equiv\mathcal S$ (homogeneous), but still update the total field $\mathbf E_i$ each iteration — and you get the **Born Iterative Method (BIM)**: cheaper (no reciprocity solves), but slower / less accurate for strong contrast, because it ignores how the object bends the *receiver* coupling. DBIM updates **both** the field and the receiver operator. The helper flips between them with `distorted=True/False` so you can see the gap yourself.

---

## 5. The DBIM algorithm

DBIM is **Gauss–Newton on the forward map**: repeatedly (i) measure how wrong the current object is *in data space*, (ii) use the Jacobian $J$ to find the object-change that would fix it, (iii) take that step. Here is every line with its math *and* its physical meaning.

$$
\begin{aligned}
&\chi_0=\mathbf 0\quad\text{(empty guess; or warm-start from the I1 Born estimate)}\\
&\textbf{for } n=0,1,2,\dots:\\
&\quad \text{(1)}\ \ \mathbf E^{(n)}_i,\ \mathbf d^{\text{sim}}=\mathcal F(\chi_n)\qquad\text{full forward per incidence (reuse MoM2D)}\\
&\quad \text{(2)}\ \ \Delta\mathbf d=\mathbf d^{\text{meas}}-\mathbf d^{\text{sim}};\quad \text{res}_n=\lVert\Delta\mathbf d\rVert/\lVert\mathbf d^{\text{meas}}\rVert\\
&\quad \text{(3)}\ \ \textbf{if } \text{res}_n<\text{tol}:\ \textbf{break}\\
&\quad \text{(4)}\ \ J=\mathcal S^{\text{dist}}(\chi_n)\,\text{diag}(\mathbf E^{(n)})\qquad\text{Jacobian at }\chi_n\ (\S 3\text{–}\S 4)\\
&\quad \text{(5)}\ \ \Delta\chi=\arg\min_{\Delta\chi}\ \lVert J\,\Delta\chi-\Delta\mathbf d\rVert^2+\mu\lVert\Delta\chi\rVert^2\qquad\text{(LSMR, damp}=\sqrt\mu)\\
&\quad \text{(6)}\ \ \chi_{n+1}=\chi_n+\gamma\,\Delta\chi\qquad\text{step }\gamma\in(0,1]
\end{aligned}
$$

**Line by line — what it means and why it's there:** 

1. **Simulate the current guess.** Run the *full* forward map $\mathcal F(\chi_n)$ (§2): solve each incidence's interior field $\mathbf E^{(n)}_i$, radiate to get the data $\mathbf d^{\text{sim}}$ this object *would* produce. *Physics:* "if the object really were $\chi_n$, here's what the receivers would see." (Keep the $\mathbf E^{(n)}_i$ — line 4 needs them.)
2. **Measure the mismatch.** $\Delta\mathbf d$ = measured minus simulated data — the part of the measurement the current guess fails to explain. $\text{res}_n$ is its relative size, the honest score.
3. **Stop if explained.** If the guess already reproduces the measurements ($\text{res}_n<\text{tol}$), the object is found — done.
4. **Build the Jacobian at the current object.** $J=\mathcal S^{\text{dist}}(\chi_n)\,\text{diag}(\mathbf E^{(n)})$ (§3), assembled by `build_frechet_operator` from the given $G_{\text{tr}}^{\text{dist}}$ (§4) and the total fields. *Physics:* the linear map "how the receiver data moves if I nudge the object by $\Delta\chi$," computed **through the current object**, not empty space.
5. **Solve the linearized problem for the update.** Find the object-nudge $\Delta\chi$ whose predicted data-change $J\,\Delta\chi$ best cancels the mismatch $\Delta\mathbf d$ — a regularized least squares, **identical to I1** but fitting the *residual* $\Delta\mathbf d$ for an *update* $\Delta\chi$ (not the full data for the whole $\chi$). Same LSMR, same `damp=√μ`. This is the Newton "solve $J\,\Delta\chi=\text{residual}$" step; $\mu$ tames $J$'s ill-conditioning.
6. **Take the step.** $\chi_{n+1}=\chi_n+\gamma\,\Delta\chi$. With $\gamma=1$ it's full Gauss–Newton; $\gamma<1$ (or backtracking) damps overshoot on strong contrast. Loop back to line 1 and re-linearize at the improved object.

*In one sentence:* keep adjusting the object until its simulated echoes match the measured echoes, and at every step use the best linear estimate (the Jacobian) of which adjustment closes the gap.

**Notes that matter:**

- **The inner solve is I1, verbatim.** Line 5 is the same regularized least-squares LSMR call as I1 — only the operator ($J$ vs $\mathbf A$) and the right-hand side ($\Delta\mathbf d$ vs $\mathbf d$) differ.
- **Regularize the update, and consider damping the step.** If $\text{res}_n$ grows, shrink $\gamma$ (backtracking: halve $\gamma$ until the residual actually drops) or raise $\mu$. Start with $\gamma=1$ and only add backtracking if it stalls.
- **Warm start.** $\chi_0=\mathbf 0$ makes the first outer step *identical to I1* (total field = incident field, $\mathcal S^{\text{dist}}=\mathcal S$; see §3.5). You may warm-start from the I1 Born estimate to save one iteration.
- **Stopping.** Stop on $\text{res}_n$ plus a `max_outer` cap. On noisy data, stop *early* (discrepancy principle) — semi-convergence again.

---

## 6. What you implement vs. what is given

**TODO (you write these):**

- `simulate_scattered_data(centers, chi, k_b, d, dS, E_inc_set, rx)` → `(d_sim, E_tot_set)`. For each incidence: `build_D` once, `solve_total_field`, `scattered_field`; stack the data, keep the total fields. This is $\mathcal F(\chi)$ (§2). Reuses F1 functions only.
- `DBIMInverter.reconstruct(data, forward=None, x0=None)` → `(chi_hat, info)`. The outer loop of §5: simulate → residual → (given) distorted operator → LSMR update → step → repeat; record `res_history`.

**GIVEN (don't need to write — study them):**

- `distorted_green_matrix(centers, chi, k_b, d, dS, rx, distorted=True)` — §4, the reciprocity computation of $G_{\text{tr}}^{\text{dist}}$ (returns homogeneous $G_{\text{tr}}$ when `distorted=False`, i.e. BIM).
- `build_frechet_operator(centers, rx, E_tot_set, k_b, dS, G_tr)` — wraps a `BornOperator`, overwriting its `G_tr` with the distorted one and passing the total fields as the "incident" set, so `matvec`/`rmatvec` become $J$/$Jᴴ$ (reuses I1).
- `make_dbim_problem(...)` — synthetic strong-scatterer problem (full-forward `physical` data), reusing I1's `make_born_problem`.
- `DBIMInverter.__init__` + registration.

---

## 7. Self-test checklist (`tests/test_i2.py`, tick them off)

- **I2.1** `simulate_scattered_data(χ_true)` == the problem's `d` (<1e-10). Guards your forward re-simulation.
- **I2.2** distorted-Born **adjoint gate**: ⟨J v,u⟩=⟨v,Jᴴu⟩ (<1e-10) on random complex vectors — make this pass before trusting any reconstruction.
- **I2.3 (headline)** DBIM vs Born: with `physical` data on a moderate scatterer, the full-forward data residual of χ̂_DBIM is a small fraction of χ̂_Born's, and its $\chi$-error is lower.
- **I2.4** `res_history` decreases overall and reaches a small value within `max_outer`.
- **I2.5** `build("inverter","dbim")` returns a working inverter.

---

## 8. Pitfalls (by probability of hitting them)

- **Forgetting to re-`build_D` each outer step.** The operator and fields must be rebuilt at the *current* $\chi$ every iteration — that's the whole idea. Caching D across iterations silently makes it BIM-without-field-update (i.e. one Born step forever).
- **Fitting the wrong thing in the inner solve.** LSMR fits $\Delta\mathbf d$ (residual) for $\Delta\chi$ (update), then you *add* to $\chi$. Fitting the full `d` for a whole $\chi$ each step is not DBIM.
- **Adjoint mismatch after substitution.** As long as you reuse `BornOperator` (whose `matvec`/`rmatvec` are constructed as exact adjoints), I2.2 passes automatically; if you hand-roll $J$, you must supply the exact $Jᴴ$.
- **Overshoot on strong contrast.** If the residual grows, add a backtracking line search on $\gamma$, or raise μ. (Don't just crank `max_outer`.)
- **Complex vs real $k_b$.** I2 assumes a **real background** wavenumber (lossless coupling medium); the object contrast $\chi$ may be complex. `AH_op`'s `conj(k_b²·dS)` is exact here. Lossy-background (complex $k_b$) hardening is a separate, later task.

---

## 9. References

- Y. M. Wang & W. C. Chew, "An iterative solution of the two-dimensional electromagnetic inverse scattering problem," *Int. J. Imaging Syst. Technol.* 1(1), 1989 — the **Born Iterative Method (BIM)**.
- W. C. Chew & Y. M. Wang, "Reconstruction of two-dimensional permittivity distribution using the distorted Born iterative method," *IEEE Trans. Med. Imaging* 9(2), 1990 — the **DBIM**.
- Standard treatments: Chew, *Waves and Fields in Inhomogeneous Media* (ch. on inverse scattering); Pastorino, *Microwave Imaging* (ch. on Newton/DBIM-type methods).

---

*I2 tutorial v1 · 2026-07-05 · "Born, in a loop, re-linearized" · pairs with [[I1_Tutorial_Born-linear-inversion]] and CODE_GUIDE Appendix E.*
