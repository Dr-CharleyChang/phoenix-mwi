---
title: "F2 Tutorial: CG-FFT — the matrix you never build"
tags: [MWI, tutorial, F2, CG-FFT, BTTB, Toeplitz]
status: in-progress
date: 2026-06-11
related: "F1_Tutorial_2D-MoM-and-Mie-validation.md"
---

# F2 Tutorial: CG-FFT acceleration of the 2D MoM forward solver

> **Goal:** make `tests/test_f2.py` (T9–T14) go green by implementing
> `mwisim/operators.py` — a **matrix-free** version of F1's forward solve.
> Same physics, same answers (to solver tolerance), but `O(N log N)` per
> iteration and `O(N)` memory instead of `O(N^3)` / `O(N^2)`.
>
> Why it matters: at `npl=15`, a full-grid F1 problem hits N≈16k and the dense
> `build_D` can't even allocate (4 GB for the distance temp alone). 3D is
> hopeless densely. CG-FFT is the standard escape hatch — and its FFT core is
> exactly what later maps onto the Zynq/Zenith-Radar pipeline.

## §1 The structural observation (do this on paper first)

Write out F1's matrix entries:

$$D_{mn} = g(\mathbf r_m - \mathbf r_n)\,\chi_n,\qquad
g(\boldsymbol\rho)=\begin{cases}
\text{pref}\cdot J_1(k_b a)\,H_0^{(2)}(k_b|\boldsymbol\rho|) & \boldsymbol\rho\neq 0\\[2pt]
\text{pref}\cdot H_1^{(2)}(k_b a)\;-\;1 & \boldsymbol\rho=0
\end{cases}$$

with $\text{pref}=-(j\pi k_b a/2)$, $a=d/\sqrt\pi$. **Note the $-1$:** the F1
self-cell lower-limit term must survive into F2 — it lives in $g(0)$.

Key fact: $g$ depends only on the **displacement** $\mathbf r_m-\mathbf r_n$.
On a regular $N_y\times N_x$ grid with row-major ordering ($n = i_y N_x + i_x$),
this makes $D$ **block-Toeplitz with Toeplitz blocks (BTTB)**. Convince yourself:
write the $3\times3$-grid $D$ and watch the diagonals repeat.

A BTTB matvec is a 2D discrete convolution:

$$(D\mathbf x)_{i_y,i_x}=\sum_{i_y',i_x'} g[i_y-i_y',\,i_x-i_x']\,\big(\chi\odot \mathbf x\big)_{i_y',i_x'}$$

Convolution ⇒ FFT. That's the whole trick. **Important order of operations:**
$\chi$ multiplies $x$ *before* the convolution (it's $\chi_n$, column index — the
*source* cell), not after.

---
### 1.1 what is BTTB and why Toeplitz

The kernel only depends on the **displacement** between two cells, not their absolute positions:
$$K_{mn} = g(\mathbf{r}_m-\mathbf{r}_n), \quad g(\rho) = g\left(d\sqrt{\Delta i_x^2+\Delta i_y^2}\right)$$
So if cell $m=(i_y,i_x)$ and cell $n=(j_y,j_x)$, then $K_{mn}=g_{\,p,q}$ with $p=i_y-j_y$, $q=i_x-j_x$. **One value per displacement** $(p,q)$, not $N^2$ independent numbers. (For now ignore $\chi$ — it's a separate per-column scaling, handled at the end.)

### 1.2 Toeplitz, then BTTB

**Toeplitz (1D):** a matrix constant along each diagonal, $T_{ij}=t_{i-j}$. It's defined by one vector, not a full matrix.

**Block-Toeplitz with Toeplitz blocks (BTTB, 2D):** partition the $N=N_yN_x$ matrix into $N_y\times N_x$ blocks of size $N_x\times N_x$.
- **Block-Toeplitz**: the block at block-position $(I,J)$ depends only on $I-J$ (Toeplitz *in the y-index*, at the block level).
- **Toeplitz blocks**: inside each block, entries depend only on $i_x-j_x$ (Toeplitz *in the x-index*).

Formally: $K_{(i_y,i_x),(j_y,j_x)}=g_{\,i_y-j_y,\;i_x-j_x}$ — outer index → block-Toeplitz, inner index → Toeplitz blocks.

### 1.3 The 3×3 grid ($N_y=N_x=3$, so $N=9$)

Row-major flat index $n=3i_y+i_x$, ordering
$(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)$.

First, one **x-Toeplitz block** for a fixed y-displacement $p$ (notice constant diagonals):

$$
B^{(p)}=\begin{pmatrix} g_{p,0} & g_{p,-1} & g_{p,-2}\\ g_{p,1} & g_{p,0} & g_{p,-1}\\ g_{p,2} & g_{p,1} & g_{p,0}\end{pmatrix}
$$

Then the full $9\times9$ in **block form** (notice the blocks themselves form a Toeplitz pattern):

$$
K=\begin{pmatrix} B^{(0)} & B^{(-1)} & B^{(-2)}\\ B^{(1)} & B^{(0)} & B^{(-1)}\\ B^{(2)} & B^{(1)} & B^{(0)}\end{pmatrix}
$$

Fully expanded (block dividers shown) — **this is the "watch the diagonals repeat" payoff**:

$$
K=\left(\begin{array}{ccc|ccc|ccc}
g_{0,0} & g_{0,-1} & g_{0,-2} & g_{-1,0} & g_{-1,-1} & g_{-1,-2} & g_{-2,0} & g_{-2,-1} & g_{-2,-2}\\
g_{0,1} & g_{0,0} & g_{0,-1} & g_{-1,1} & g_{-1,0} & g_{-1,-1} & g_{-2,1} & g_{-2,0} & g_{-2,-1}\\
g_{0,2} & g_{0,1} & g_{0,0} & g_{-1,2} & g_{-1,1} & g_{-1,0} & g_{-2,2} & g_{-2,1} & g_{-2,0}\\
\hline
g_{1,0} & g_{1,-1} & g_{1,-2} & g_{0,0} & g_{0,-1} & g_{0,-2} & g_{-1,0} & g_{-1,-1} & g_{-1,-2}\\
g_{1,1} & g_{1,0} & g_{1,-1} & g_{0,1} & g_{0,0} & g_{0,-1} & g_{-1,1} & g_{-1,0} & g_{-1,-1}\\
g_{1,2} & g_{1,1} & g_{1,0} & g_{0,2} & g_{0,1} & g_{0,0} & g_{-1,2} & g_{-1,1} & g_{-1,0}\\
\hline
g_{2,0} & g_{2,-1} & g_{2,-2} & g_{1,0} & g_{1,-1} & g_{1,-2} & g_{0,0} & g_{0,-1} & g_{0,-2}\\
g_{2,1} & g_{2,0} & g_{2,-1} & g_{1,1} & g_{1,0} & g_{1,-1} & g_{0,1} & g_{0,0} & g_{0,-1}\\
g_{2,2} & g_{2,1} & g_{2,0} & g_{1,2} & g_{1,1} & g_{1,0} & g_{0,2} & g_{0,1} & g_{0,0}
\end{array}\right)
$$

Two patterns to verify by eye:
1. **Block level**: the block-diagonal is all $B^{(0)}$, the block super-diagonal all $B^{(-1)}$, sub-diagonal all $B^{(1)}$ — i.e. blocks repeat along block-diagonals = block-Toeplitz.
2. **Inside each block**: every diagonal is constant ($g_{p,0}$ on the diagonal, $g_{p,-1}$ one above, etc.) = Toeplitz blocks.

The entire $9\times9$ is generated by just the $(2N_y-1)\times(2N_x-1)=5\times5$ table of $g_{p,q}$ values ($p,q\in\{-2,-1,0,1,2\}$) — that's the whole "kernel".

### Why this is the whole F2 idea

- A **BTTB matrix times a vector = 2D convolution** of the kernel $g_{p,q}$ with the (reshaped $N_y\times N_x$) vector. Convolution → FFT → $O(N\log N)$ instead of $O(N^2)$, and you store the $5\times5$ kernel instead of the $9\times9$ matrix.
- **Symmetry bonus**: $\rho$ only sees magnitudes, so $g_{p,q}=g_{|p|,|q|}$ ⇒ $B^{(p)}=B^{(-p)}$ and each block is symmetric. Nice, but not needed for the FFT.
- **Where $\chi$ goes**: $D_{mn}=K_{mn}\,\chi_n$ — a per-column (source-cell) scaling, which is *not* BTTB. But $(Dx)_m=\sum_n g_{m-n}\,(\chi_n x_n)$, so you **multiply $x$ by $\chi$ first** (cheap elementwise), then apply the BTTB convolution $K$. That's exactly why `apply_D` says "multiply by chi FIRST, then convolve." The FFT accelerates the $K$ part; $\chi$ is a free diagonal pre-multiply.

So your `GreenFFT.__init__` builds that $5\times5$-style kernel (in general $(2N_y{-}1)\times(2N_x{-}1)$), zero-pads/embeds it into a circulant, and FFTs it once; `_conv` does the convolution; `apply_D` adds the $\chi$ pre-multiply. The 9×9 above is the thing you're *never going to form* — you only ever touch its generating kernel.

### 1.4 From BTTB to a 2D FFT — the mechanics, worked end to end

This is the step that "K is BTTB → use FFT" usually hand-waves. Here it is in full.

**Step A — the matvec *is* a 2D linear convolution.** Reshape the flat vectors back
to grids, $X[j_y,j_x]$ and $Y=Kx$ as $Y[i_y,i_x]$. Then

$$
Y[i_y,i_x]=\sum_{j_y,j_x} g[\,i_y-j_y,\;i_x-j_x\,]\,X[j_y,j_x]
\;=\;(g * X)[i_y,i_x]
$$

That double sum is the textbook definition of a **2D discrete convolution** of the
kernel $g$ with the field $X$ (the matrix index $i-j$ *is* the convolution shift).
A 1D Toeplitz matvec is a 1D convolution; BTTB is its 2D big brother.

**Step B — the convolution theorem (why FFT enters at all).** For length-$P$
**periodic** (circular) sequences, the DFT turns convolution into a plain
element-wise product:

$$
\text{DFT}\{g \circledast x\} = \text{DFT}\{g\}\cdot\text{DFT}\{x\}
\quad\Longrightarrow\quad
g \circledast x = \text{IFFT}\big(\text{FFT}(g)\cdot\text{FFT}(x)\big)
$$

Intuition: each Fourier mode is an **eigenvector** of circular convolution, so in the
frequency domain the operator is just a diagonal (one complex scale per mode). 2D is
identical with `fft2`. Cost: two FFTs ($O(P\log P)$) + one element-wise multiply
($O(P)$) — versus $O(N^2)$ for the dense matvec.

**Step C — the catch: FFT gives *circular*, we need *linear*.** Our matvec
$\sum_j g[i-j]x[j]$ is a **linear** convolution (no wrap). `fft2` computes the
**circular** one ($i-j$ taken mod $P$), where the kernel illegally "sees" sources from
the opposite edge. They only agree if you **pad** so the wrap-around lands in empty
space. That padding is the circulant embedding of §2.

**Worked 1D example (do this by hand once — it removes all mystery).** Take $N=3$, a
1D Toeplitz $y_i=\sum_{j=0}^{2} g_{i-j}x_j$, kernel displacements $g_{-2}\dots g_{2}$:

$$
y_0=g_0x_0+g_{-1}x_1+g_{-2}x_2,\quad
y_1=g_1x_0+g_0x_1+g_{-1}x_2,\quad
y_2=g_2x_0+g_1x_1+g_0x_2
$$

Pad to $P=2N-1=5$. Build the circulant's first column by placing displacement $p$ at
index $p\bmod 5$ — non-negatives at the bottom, **negatives wrap to the top**:

$$
c=[\,g_0,\;g_1,\;g_2,\;g_{-2},\;g_{-1}\,]\quad(\text{indices }0,1,2,3,4;\;3=-2\bmod5,\;4=-1\bmod5)
$$

Zero-pad the field: $x_p=[x_0,x_1,x_2,0,0]$. Now circular-convolve via FFT,
$y_p=\text{IFFT}(\text{FFT}(c)\cdot\text{FFT}(x_p))$, and read off:

$$
y_p[0]=c_0x_0+c_{4}x_1+c_{3}x_2=g_0x_0+g_{-1}x_1+g_{-2}x_2=y_0\;\checkmark
$$

($c_4=g_{-1}$, $c_3=g_{-2}$ — the wrapped negatives land exactly where the linear
formula wants them.) Likewise $y_p[1]=y_1$, $y_p[2]=y_2$; entries $y_p[3],y_p[4]$ are
the wrap-around **garbage you crop**. So: pad → place kernel with $\bmod P$ → zero-pad
field → FFT-multiply-IFFT → keep `[:N]`. **2D is this on each axis independently.**

**Step D — the full 2D matvec algorithm (what `_conv` does).** Precompute once:
$P_y,P_x=\texttt{next\_fast\_len}(2N-1)$ per axis; scatter $g[p,q]$ to
$(p\bmod P_y,\;q\bmod P_x)$; $\hat G=\texttt{fft2}(g_\text{pad})$. Then **every** matvec:

```
Xpad = zeros((Py, Px)); Xpad[:Ny,:Nx] = X        # zero-pad field
Y    = ifft2( G_hat * fft2(Xpad) )[:Ny, :Nx]      # convolve + crop
```

Two FFTs of size $\sim4N$ per matvec → $O(N\log N)$, memory $O(N)$. You never build the
$N\times N$ matrix.

### 1.5 From a fast matvec to the actual $E_s$ estimation

Be precise about *what* the FFT accelerates — it's a common confusion:

1. **FFT = fast matvec, NOT a fast inverse.** The expensive job is *solving*
   $(\mathbf I-\mathbf D)\mathbf E_{\text{tot}}=\mathbf E^{\text{inc}}$. But $(\mathbf I-\mathbf D)$
   is **not** a convolution — the $\mathbf I$ and the per-column $\chi$ break the
   displacement-only structure — so you cannot invert it with one FFT.
2. **Krylov does the solving; FFT does each step.** Hand $(\mathbf I-\mathbf D)$ to an
   iterative solver (BiCGStab/GMRES) that only ever needs matrix–vector products. Each
   product is $x-\texttt{apply\_D}(x)$, and `apply_D` = ($\chi$ pre-multiply) + (one
   FFT convolution). So total cost = (iteration count) $\times\,O(N\log N)$. That's the
   `apply_IminusD` → `solve_total_field` chain in your stub.
3. **Then $E_s$ is the cheap tail.** Once $\mathbf E_{\text{tot}}$ is solved, the
   scattered field at the receivers is the separate F1 step
   $E_s(\mathbf r_r)=k_b^2\,\Delta S\sum_n G(\mathbf r_r,\mathbf r_n)\chi_n E_{\text{tot},n}$
   — a small dense $M\times N$ product ($M$ = a handful of receivers on a ring, not a
   grid), which is negligible next to the in-domain solve. So **F2 accelerates the
   bottleneck (solving for $E_{\text{tot}}$); $E_s$ rides along for free.**

> **Mental model for the whole solve:** the iterative solver walks toward
> $\mathbf E_{\text{tot}}$; at each step it needs "what does $\mathbf D$ do to my current
> guess?", and instead of a giant matrix multiply it answers that question with two
> FFTs. The physics (answer) is identical to F1's direct solve; only the *how* changed.

---

## §2 Circulant embedding — the one place you can silently go wrong

`fft2` gives **circular** convolution: indices wrap modulo the grid size, so the
kernel "sees" sources from the opposite edge. The cure:

1. Pad both kernel and field to $P \ge 2N-1$ per axis
   (use `scipy.fft.next_fast_len(2*n-1)` — a fast composite length).
2. The kernel must hold **all** displacements $d\in[-(N-1),\,N-1]$ per axis.
   Place displacement $d$ at padded index $d \bmod P$ — negative displacements
   wrap to the high end (`np.mod` does this; `np.ix_` scatters the 2D block).
3. Pad the field with zeros into the top-left $N_y\times N_x$ corner.
4. After `ifft2`, slice `[:Ny, :Nx]` back out. The aliased garbage lives in the
   padding you discard.

If you skip or botch the padding, the error is *small but mesh-dependent* — it
looks like discretization error and survives casual inspection. Only the
random-vector test against dense `build_D` (T9) catches it instantly.

## §3 What to implement (`mwisim/operators.py`)

| Stub | Hints |
|---|---|
| `GreenFFT.__init__` | recover `(ny, nx)` (helper given); build displacement grids `np.arange(-(n-1), n)`; `rho = d*sqrt(dx²+dy²)`; evaluate `g` (off-diag + self at `rho==0` — `np.where`, wrap Hankel call in `np.errstate(invalid="ignore")` since `hankel2(0,0)` is NaN); zero-pad per §2; store `fft2(g_pad)` **once** |
| `_conv` | zero-pad field → `ifft2(G_hat * fft2(vp))` → slice `[:ny,:nx]` |
| `apply_D` | reshape flat `x` to grid, multiply by `chi` **first**, convolve, ravel |
| `apply_IminusD` | `x - apply_D(x)` |
| `solve_total_field` | wrap `as_linear_operator()` (given) in `scipy.sparse.linalg.bicgstab` / `gmres`. Gotchas: SciPy ≥1.12 uses `rtol` not `tol` (try/except TypeError); gmres wants explicit `callback_type="pr_norm"`; count iterations with a callback; report final relative residual yourself — don't trust `status` alone |

Conventions to preserve: row-major ravel (axis 0 = y!), `e^{+jωt}`/`H^(2)`,
and the self-cell `-1`.

## §4 Validation strategy (the F1 discipline, sharpened)

**Never benchmark before the fast path equals the slow path.**

1. **T9 first, on a random vector.** Random vectors exercise every matrix entry;
   physical fields don't. Target: `rel_l2_error(op.apply_D(x), D @ x) < 1e-12`
   (you should see ~1e-16). If you're at 1e-2: ravel orientation or chi-side bug.
   If only the diagonal disagrees: you lost the `-1`.
2. **T10/T11/T12**: `(I-D)x` matches; BiCGStab and GMRES reproduce the F1 direct
   solve to the iterative tolerance.
3. **T13 end-to-end**: scattered field through the fast path still matches Mie
   (<5%) — F2 must change the *how*, never the physics.
4. **Then** benchmark (`scripts/run_f2.py`): time + memory vs N, dense vs FFT.
   Watch the **iteration count** too: flat-in-N iters ⇒ true O(N log N);
   growing iters ⇒ you need a preconditioner (expected at high contrast).

## §5 Predicted traps (check these when stuck)

- **Transposed kernel axes**: weak scatter still looks fine (D ≈ diagonal),
  strong scatter quietly diverges. The F1 lesson reincarnated — T9's random
  vector pins it.
- **chi on the wrong side**: `conv(g, chi*x)` ✔ vs `chi*conv(g, x)` ✘ —
  $\chi_n$ indexes the source (column), not the observer (row).
- **`hankel2(0, 0)` = NaN** contaminating the whole kernel: evaluate off-diagonal
  with `errstate`, then `np.where(rho > 0, g_off, g_self)`.
- **Padding only to N** (not 2N−1): wrap-around aliasing, see §2.
- **Stale `.pyc` shadowing your edits** on a mounted/odd filesystem: if a fix
  "isn't taking", run with `PYTHONPYCACHEPREFIX=/tmp/x` or purge `__pycache__`.

## §6 Self-test checklist (your progress bar)

```
pytest tests/test_f2.py -q
```

- [ ] T9  apply_D == dense build_D @ x (ε_r = 2 and 8, random vector, <1e-12)
- [ ] T10 (I−D)x matches dense
- [ ] T11 BiCGStab matches direct solve (<1e-7)
- [ ] T12 GMRES matches direct solve (<1e-7)
- [ ] T13 end-to-end scattered field: fast == slow, and both == Mie (<5%)
- [ ] T14 infer_grid_shape round-trips and rejects ragged grids (given helper —
      read it, it encodes the ravel convention)

When 15/15 (F1+F2) are green, run `python scripts/run_f2.py` for the scaling
figure, then write `docs/F2_milestone.md` with *your* numbers and war stories
(a reference draft from a prior AI pass is kept as
`F2_milestone_DRAFT_reference.md` — compare, don't copy).

---

*A complete reference implementation exists in git history (`529ab9d`) — diff
against it after you're green, not before.*
