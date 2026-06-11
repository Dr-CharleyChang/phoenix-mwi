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
