---
title: "Phase 0 Milestone: the platform spine (interfaces + registry + faithful refactor)"
tags: [phoenix-mwi, milestone, phase0, architecture, ABC, registry, build-journal]
status: milestone — Phase 0 complete
date: 2026-06-17
related: "[[PROJECT_PLAN]], [[CODE_GUIDE_codebase-and-algorithm-from-zero]], [[F2_milestone]]"
---

# Phase 0 Milestone — the platform spine

> **In one line:** turned the F1/F2 *scripts* into the skeleton of a *platform*. Added the abstract interfaces (ABCs) every layer must fit, a name→class registry, and a GPU/CPU backend hook — then refactored the validated 2D MoM forward solver to run *through* those interfaces without changing a line of physics. `pytest`: **22 passed** (15 physics + 7 Phase-0). The ambition now lives in the contracts; the execution stays one slice at a time.

## 1. What was built

| File                           | Role                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `mwisim/core/interfaces.py`    | 9 ABCs: `Phantom`, `SceneBuilder`, `ForwardSolver`, `DataSource`, `Preprocessor`, `Imager`, `Inverter`, `Reconstructor`, `Evaluator` |
| `mwisim/core/registry.py`      | `register` / `build` / `available` — optional name-based selection (gprMax-style)                                                    |
| `mwisim/core/backend.py`       | `xp` array-module hook (numpy now; CuPy/PyTorch GPU later) — zero-cost neutrality                                                    |
| `mwisim/phantoms/circle.py`    | `CirclePhantom(Phantom)` — wraps F1 `grid`+`assign_contrast`                                                                         |
| `mwisim/forward/mom2d.py`      | `MoM2D(ForwardSolver)` — wraps `mom` (dense) + `operators.GreenFFT` (cgfft), backend selectable by a constructor knob                |
| `mwisim/evaluation/metrics.py` | `RelL2Evaluator(Evaluator)` — wraps `rel_l2_error`                                                                                   |
| `tests/test_phase0.py`         | P1–P6                                                                                                                                |

The adapters are **thin**: they import and reuse the already-validated functions and re-express them as the platform contract. The physics modules (`grid`, `green`, `mom`, `mie`, `metrics`, `operators`) were not touched.

## 2. Validation results

`tests/test_phase0.py`: **7 passed** (P5 is parametrized ×2). Full suite: **22 passed**.

- **P1** — ABC enforcement: an abstract interface can't be instantiated; a subclass missing a method can't either; a complete subclass can.
- **P2** — registry roundtrip: `@register` files a class, `build` retrieves+instantiates it, duplicate names are rejected.
- **P3** — `CirclePhantom.grid()/contrast()` reproduce the raw `make_grid`/`assign_contrast`.
- **P4** — `MoM2D(method="dense")` reproduces the raw F1 pipeline to **<1e-12**, and its scattered field still matches the analytic **Mie** series **<5%**.
- **P5** — `MoM2D(method="cgfft")` matches `method="dense"` to **<1e-7** at $\varepsilon_r$ = 2 and 8.
- **P6** — `build("forward","mom2d")` / `build("phantom","circle")` produce working objects.

The point of P3–P5 is a *faithfulness* guarantee: the refactor changed the call structure, not the numbers.

## 3. Design decisions (the durable ones)

- **Ambition in the interfaces, not the feature count.** All nine ABCs are designed for the *full* vision (3D, real data, AI). Phase 0 implements only the 2D forward slice through them; later work is "implement the interface and plug in."
- **Subclassing is the first-class extension path; the registry is optional sugar.** A user adds an algorithm by subclassing an ABC and passing the instance. `@register` + names exist only for config-/CLI-driven selection.
- **Backend neutrality is designed now, implemented later.** `core/backend.py` exposes `xp` so kernels never hard-code `numpy`; the CUDA/PyTorch backend drops in without touching call sites (PROJECT_PLAN §11.4).
- **Refactor under test cover.** Wrapping validated code behind an interface is only safe because the 15 physics tests guard the underlying functions — the same discipline that let F1→F2 swap the solver. Tests are the specification.

## 4. What it unlocks

The `Inverter` and `Evaluator` contracts (plus the `A_op`/`AH_op` Born operators already in `operators.py`) are exactly what the inversion stage plugs into. Next is **I1 — Born linear inversion** (the first χ-map / first image), implemented as an `Inverter` behind this spine.

## 5. Still open in the spine (lands early in Phase 1)

- The unified **data schema** (so synthetic and real measurements are interchangeable).
- The declarative **scene file + `SceneBuilder`** implementation (gprMax-style, no-Python problem definition).
- The `Pipeline` and `Reporter` concretes (orchestration + text/figure reporting).

---

*Phase 0 closed 2026-06-17 · "design the skeleton for the whole animal, grow one limb at a time" · 22/22, physics unchanged.*
