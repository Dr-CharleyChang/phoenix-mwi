---
title: "F2 milestone: CG-FFT acceleration of the 2D MoM forward solver (passed)"
tags: [MWI, milestone, F2, CG-FFT, BTTB, Toeplitz, build-journal]
status: done
date: 2026-06-11
related: "F1_milestone.md"
---

# F2 milestone: from O(N³) dense to matrix-free CG-FFT

> **In one line:** rewrote the F1 dense MoM solve as a **matrix-free** operator —
> the coupling matrix is block-Toeplitz, so `D @ x` is a 2D FFT convolution.
> Validated against dense `build_D` to **machine precision (5e-16)** and against
> the F1 direct solve to **~1e-8**. Result: **N = 102,400 solved in 0.24 s** where
> the dense matrix alone would need **168 GB**. `pytest 15/15`.

## 1. The structural insight

On a regular grid the Richmond MoM matrix entries

$$D_{mn} = g(\mathbf r_m - \mathbf r_n)\,\chi_n$$

depend only on the **displacement** $\mathbf r_m - \mathbf r_n$, never on absolute
position. Order the cells row-major and $D$ becomes **block-Toeplitz with Toeplitz
blocks (BTTB)**. A BTTB matrix–vector product is exactly a **2D convolution** of the
first-column kernel $g$ with the (contrast-weighted) field:

$$(D\mathbf x)_{i,j} = \sum_{i',j'} g[\,i-i',\,j-j'\,]\,(\chi\mathbf x)_{i',j'}.$$

Convolutions are FFTs: $O(N\log N)$ per matvec, $O(N)$ storage — versus the dense
$O(N^2)$ build / $O(N^3)$ factorisation. The kernel reproduces `build_D` exactly,
**including the F1 self-cell fix**: $g(0)=\text{pref}\cdot H_1^{(2)}(k_b a)-1$, the
"$-1$" being the missing lower-limit term we hunted down in F1 §3.

## 2. What was built (`mwisim/operators.py`)

| Piece | What it does |
|---|---|
| `infer_grid_shape` | recover $(N_y,N_x)$ from the raveled `centers` (row-major) |
| `GreenFFT.__init__` | precompute kernel $g$ over all displacements, **circulant-embed**, store `fft2(g_pad)` once |
| `GreenFFT.apply_D` | matrix-free $D\mathbf x = \text{conv}(g,\ \chi\odot\mathbf x)$ — two FFTs, never forms $N\times N$ |
| `GreenFFT.apply_IminusD` | the forward-solve operator $(\mathbf I-\mathbf D)\mathbf x$ |
| `GreenFFT.solve_total_field` | wrap as a SciPy `LinearOperator`, solve with **BiCGStab** (classic CG-FFT pairing) or **GMRES** |

### The one subtlety that bites: circulant embedding

A naive FFT gives **circular** convolution — it wraps the kernel around the grid and
contaminates the answer (aliasing). The fix is to zero-pad both kernel and field to
size $\ge 2N-1$ per axis before the FFT, so the wrap-around lands in the discarded
padding. We pad to `scipy.fft.next_fast_len(2N-1)` for speed, place displacement $d$
at index $d \bmod P$ (negative displacements wrap to the high end), and slice
`[:Ny,:Nx]` back out. Get the padding wrong and the error is small but **nonzero and
mesh-dependent** — exactly the kind of bug that hides until a convergence study.

## 3. Validation results

- **Operator exactness** (T9, T10): `apply_D` vs dense `build_D @ x` →
  rel error **5.0e-16** at both $\varepsilon_r=2$ and $\varepsilon_r=8$. The
  matrix-free path is the dense path, to the last bit.
- **Solve agreement** (T11, T12): BiCGStab and GMRES vs the F1 direct solve →
  **~1e-8** (= the iterative tolerance), weak and strong contrast.
- **End-to-end** (T13): scattered field through the fast path still matches the
  analytic **Mie** series (< 5%) — F2 changes the *how*, not the physics.
- **Tests**: `pytest -q` → **15 passed** (F1's 7 + F2's T9–T14).

### Benchmark (`scripts/run_f2.py` → `docs/fig_f2_scaling.png`)

| N | BiCGStab iters | t (CG-FFT) | t (dense) | mem (CG-FFT) | mem (dense $D$) |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 7 | 0.008 s | 7.9 s | 0.5 MB | 268 MB |
| 6,400 | 7 | 0.013 s | 22.4 s | 0.8 MB | 655 MB |
| 25,600 | 7 | 0.051 s | — | 3.3 MB | 10.5 GB |
| 102,400 | 7 | 0.24 s | — | 13 MB | **168 GB** |

At N = 6,400 the matrix-free solve is **~1700× faster** and **800× smaller**; beyond
N ≈ 25k the dense matrix simply does not fit in RAM, while CG-FFT stays under 15 MB.

> **Why iteration count stays flat at 7** (weak scatter $\varepsilon_r=2$): the
> spectrum of $(\mathbf I-\mathbf D)$ is clustered near 1, so Krylov convergence is
> grid-independent. Strong scatterers ($\varepsilon_r=8$) need ~28 BiCGStab iters —
> still $O(1)$ in $N$ — which motivates a preconditioner before 3D / very high
> contrast.

## 4. Debug war story: trust nothing matrix-free until it equals the matrix

The discipline that made F2 safe was refusing to benchmark before proving
correctness. The plan was strict: **first** make `apply_D(x)` equal `build_D @ x` to
machine precision on a *random* vector (not a physical one — random vectors exercise
every matrix entry, physical fields don't), **then** swap in the iterative solver,
**then** time it. Two traps this caught:

1. **Ravel/displacement orientation.** `make_grid` ravels row-major
   ($n=i_y N_x + i_x$), so the convolution's first axis must be $i_y$ and the kernel
   must be indexed $g[i_y-i_y',\,i_x-i_x']$. Transpose it and weak scatter still looks
   fine (near-diagonal $D$) while strong scatter quietly diverges — the F1 lesson,
   reincarnated. The random-vector test against dense pins it instantly.
2. **The self-cell `-1`.** The FFT kernel must carry F1's corrected self term. Drop it
   and `apply_D` disagrees with `build_D` by exactly $\chi_n x_n$ on the diagonal —
   visible immediately at 5e-16-vs-O(1), invisible in a convergence plot.

(Process footnote, not physics: a stale, OS-locked `.pyc` in the working mount kept
shadowing edited source during testing — worth a `PYTHONPYCACHEPREFIX=/tmp/...` or a
cache purge when "the fix isn't taking" on a networked/!mounted filesystem.)

## 5. Transferable lessons

1. **Exploit structure before you optimise.** The 10³× win came from *recognising*
   BTTB, not from faster hardware. Always ask "what does my operator's matrix look
   like?" before reaching for more cores.
2. **Validate the fast path against the slow path, bit-for-bit, on random inputs.**
   Machine-precision agreement on a random vector is a far stronger guarantee than
   "the picture looks right".
3. **Zero-pad to kill circular-convolution aliasing.** The $2N-1$ embedding is the
   whole ballgame for FFT-based operators; it generalises directly to 3D ($2N-1$ per
   axis) and to the inversion $A/A^H$ operators.
4. **Iteration count, not just per-iteration cost, sets the scaling.** Flat iters ⇒
   true $O(N\log N)$; growing iters ⇒ you need a preconditioner. Watch both.

## 6. Next steps

- **F2.1 (optional)**: a simple diagonal / two-grid preconditioner to flatten the
  strong-contrast iteration count, and an FFT-based $A/A^H$ for the in-domain Green
  action (reused directly by DBIM).
- **I1–I4**: inversion (Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM). The matrix-free
  `A_op`/`AH_op` now have a fast in-domain backbone to build on.
- **F3**: UWCEM phantom + Cole–Cole multi-frequency — realistic tissue.
- **HLS**: the FFT convolution core is exactly what maps onto the Zynq-7020 /
  Zenith-Radar FFT pipeline.

---

*F2 closed 2026-06-11 · "the matrix you never build" · dense `build_D` retained as the
ground-truth oracle, every matvec checked against it.*
