---
title: "F1 里程碑：2D MoM 正演器 + Mie 解析验证（已通过）"
tags: [MWI, milestone, F1, MoM, Mie, build-journal]
status: ✅ done
date: 2026-06-10
related: "[[F1_Tutorial_2D-MoM正演与Mie验证]]"
---

# F1 里程碑：第一个解析验证过的 2D MWI 正演器

> **一句话**：从零写出一个 2D TM 微波散射正演器（Richmond MoM / Lippmann–Schwinger），并用 Mie 级数解析解严格验证——逐点误差 3.15%、收敛曲线单调下降、`pytest 7/7`。这是整个 MWI 项目的地基。

## 1. 做了什么

实现了一条完整的"正演 + 验证"流水线（`mwisim/`）：

| 模块 | 函数 | 干什么 |
|---|---|---|
| `grid.py` | `make_grid`, `assign_contrast` | 方形域中点网格 + 对比度 $\chi$ |
| `green.py` | `green_2d` | 2D 自由空间 Green 函数 $\frac{1}{4j}H_0^{(2)}(k_bR)$ |
| `mom.py` | `build_D`, `incident_plane_wave`, `solve_total_field`, `scattered_field` | Richmond MoM 离散、平面波激励、解 $(\mathbf I-\mathbf D)\mathbf E=\mathbf E_{inc}$、算接收散射场 |
| `mie.py` | `mie_an`, `mie_scattered` | 介质圆柱散射的解析级数（真值） |
| `metrics.py` | `rel_l2_error`, `convergence_study` | 误差度量 + 网格加密研究 |

物理设定：平面波 $e^{-jk_bx}$ 照射介质圆柱（$\varepsilon_r=2\sim8$），$e^{j\omega t}$/$H^{(2)}$ 约定全程一致。

## 2. 验证结果

- **逐点对比**（`fig_pointwise.png`）：接收圈上 MoM（点）压在 Mie（线）上，实部虚部都吻合，相对 $L_2$ 误差 **3.15%**（弱散射 / `npl=15`）。
- **收敛曲线**（`fig_convergence.png`）：误差随"每波长格数"单调下降，log-log 斜率 $\approx1\sim2$——证明剩余误差是离散化、随加密消失，不是 bug。
- **测试**：`pytest -q` → **7 passed**，覆盖弱散射 sanity（T4）、Mie 自收敛（T5）、MoM-vs-Mie 弱/强散射（T6 $\varepsilon_r=2$ / T8 $\varepsilon_r=8$）。

## 3. Debug 战记：self-cell 漏项（AI 错了，物理对了）

> 这一段值得单独记，因为它是这次最有价值的教训。

**症状**：弱散射全过，但 $\varepsilon_r=2/8$ 时 MoM 和 Mie 对不上，误差 ~0.8–1.1。

**定位三板斧**：
1. **不是约定问题**——试了共轭/反号/复缩放,全更差,排除了"$H^{(2)}\leftrightarrow H^{(1)}$ 翻转"这类全局错误。
2. **加密网格,误差卡在 0.80 不降**——说明 MoM 自洽收敛到一个*稳定的错答案*,问题在公式而非分辨率。
3. **弱散射极限三方对照**(MoM / Mie / **Born**)——$\varepsilon_r\to1.01$ 时三者全一致(误差 1.6–1.9%)。这把 bug 锁死在"只在强散射现身"的地方 = **对角线 self 项**(它只在多次散射主导时起作用)。

**根因**：self-cell 积分 $\int_0^a uH_0^{(2)}(k_bu)\,du=\big[uH_1^{(2)}(u)\big]_0^{k_ba}$ 的**下限**被当成 0 漏掉了。其实 $\lim_{u\to0}uH_1^{(2)}(u)=\frac{2j}{\pi}\neq0$（$Y_1$ 在原点发散）。补回后正确自项是

$$D_{nn}=-\chi_n\Big[\tfrac{j\pi k_ba}{2}H_1^{(2)}(k_ba)+\underbrace{1}_{\text{漏了的项}}\Big]$$

代码修复就一处：`np.fill_diagonal(D, pref*hankel2(1,k_b*a) - 1)`。改完 **7/7 全绿**。

**为什么弱散射看不出**：弱散射时 $\mathbf D$ 整体很小,$(\mathbf I-\mathbf D)\approx\mathbf I$,对角线那点误差被淹没;强散射时多次散射放大它 → 反演崩。

## 4. 经验教训（可迁移）

1. **金标准必须独立、且先做对**。坚持"先把 Mie 验到自收敛、再拿它验 MoM",否则两个都错根本发现不了。
2. **弱极限三方对照是定位 bug 的杀手锏**。Born 近似无歧义,用它当第三方裁判,一步排除"全局约定错"vs"局部公式错"。
3. **收敛曲线不只是成果图,也是诊断仪**。误差"卡住不降"立刻指向公式 bug,而非分辨率不足。
4. **2D MoM 的 self-cell 下限奇异项是经典坑**（$Y_1$ 原点发散）。这个直觉对将来写 3D Green、写迭代求解器预条件都用得上。
5. **AI 给的公式也要验**。这次正是 tutorial 公式漏项,靠物理诊断抓出来——"AI 写代码,人验物理"。

## 5. 下一步

- **F2**：CG-FFT 加速（matrix-free + Toeplitz-FFT），为大规模 / 3D 铺路，且 FFT 核与 Zenith-Radar 合流。
- **I1–I4**：反演（Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM）——从"正演"迈到"成像"。
- **F3**：UWCEM 体模 + Cole-Cole 多频，贴近真实组织。

---

*F1 closed 2026-06-10 · 从"稀里糊涂"到"解析验证过的正演器" · 每个 bug 都记下来了。*
