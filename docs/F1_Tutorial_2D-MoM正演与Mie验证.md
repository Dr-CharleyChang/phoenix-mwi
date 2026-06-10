---
title: "F1 教程：2D TM 散射的 MoM/L-S 正演 + Mie 解析验证"
tags: [MWI, MoM, Richmond, Lippmann-Schwinger, Mie, forward-solver, 2D-TM, 验证]
status: tutorial v1.0
date: 2026-06-07
related: "[[Fear 2013 微波乳腺成像：全综合主笔记]] · [[BIM 正向求解器]] · [[第6章 CG]]"
---

# F1 教程：2D TM 散射的 MoM/L-S 正演 + Mie 解析验证

> [!info] 这份教程怎么用
> 先**通读一遍**建立全局图，再**自己动手实现**，卡住回头查对应小节。文档给的是公式、原理、函数清单和自测点——**不给完整代码**，留白给你练。做完你会得到整个项目的第一张硬核成果图：**散射场误差 vs 网格密度的收敛曲线**。

---

## 0. F1 的目标与验收标准

**目标**：写一个 2D TM 正演器，对"介质圆柱被平面波照射"算出散射场，并用 **Mie 级数解析解**验证它算得对。

**为什么是这个题**：
- 圆柱散射有**闭式解析解**（Mie 级数），是不依赖任何体模/实测的"真值"；
- 2D 把所有积木（Green 函数、self-cell 奇异、MoM 离散、线性求解）都练一遍，3D 只是换 Green 核、升一维，框架复用；
- 验证逻辑是"数值 vs 解析"，纯仿真就能完成，正是你现在的条件。

**验收标准（做到才算 F1 通过）**：
1. 固定网格，散射场与 Mie 的相对 $L_2$ 误差 $< 5\%$（弱散射时可 $<1\%$）；
2. 加密网格，误差**单调下降**，log-log 上呈现明确收敛阶（斜率约 $1\sim2$）；
3. 收敛曲线图 + 一张散射场实部/虚部 数值 vs 解析 的叠加对比图。

---

## 1. 物理设定：2D TM、平面波、对比度

![[Pasted image 20260608095132.png]]

**TM 极化（E 沿 $z$）**：场只有一个标量分量 $E_z(x,y)$，问题从矢量退化成**标量**——这是 2D 入门选 TM 的原因。

- **背景**：均匀无界介质，波数 $k_b=\omega\sqrt{\mu_0\varepsilon_0\tilde\varepsilon_b}$（真空时 $k_b=k_0=\omega/c$）。
- **散射体**：半径 $R_{\text{cyl}}$ 的介质圆柱，相对介电常数 $\varepsilon_r$（内部波数 $k_1=k_0\sqrt{\varepsilon_r}$）。
- **对比度函数**：$\chi(\mathbf r)=\dfrac{\tilde\varepsilon_r(\mathbf r)-\tilde\varepsilon_b}{\tilde\varepsilon_b}$，圆柱内 $\chi=\varepsilon_r/\varepsilon_b-1$，圆柱外 $\chi=0$。
- **入射场**：平面波 $E_z^{\text{inc}}(\mathbf r)=E_0\,e^{-jk_b x}$（沿 $+x$ 传播，$e^{j\omega t}$ 工程约定）。

> [!warning] 时谐约定先钉死
> 全程用 $e^{j\omega t}$（IEEE 工程约定，与你 Cole-Cole 笔记 $\tilde\varepsilon=\varepsilon-j\sigma/\omega\varepsilon_0$ 一致）。这决定了：外向波 $\sim e^{-jk_bR}$、2D Green 用 $H_0^{(2)}$、Mie 散射用 $H_n^{(2)}$。**只要 Green、入射场、Mie 三处约定一致即可；不一致会导致虚部符号翻转、散射能量变负、和解析解永远对不上。** 若你习惯物理约定 $e^{-i\omega t}$，把所有 $H^{(2)}\to H^{(1)}$、指数 $-j\to +i$ 即可。

在实际编程实现中，背景介质中的波数（假设非磁性介质，$\mu_r=1$）：
$$\begin{align}
k_b &= \omega\sqrt{\mu_0\varepsilon_0\,\varepsilon_{r,b}}=2\pi f\underbrace{\sqrt{\mu_0\varepsilon_0}}_{1/c_0}\sqrt{\varepsilon_{r,b}} \\
&= 2\pi\frac{f}{c_0}\sqrt{\varepsilon_{r,b}} = 2\pi/\lambda_0\sqrt{\varepsilon_{r,b}}
\end{align}$$
其中 $\varepsilon_{r,b}$ 就是代码里的 `eps_b`（相对介电常数）。这正是代码 [run_f1.py:L45-L46](file:///c:/Projects/Project_Pheonix/mwi/scripts/run_f1.py#L45-L46) 的形式：

```python
k_b = 2 * np.pi / lam0 * np.sqrt(P["eps_b"])
k_1 = 2 * np.pi / lam0 * np.sqrt(P["eps_r"])
```

这里 `lam0 = C0 / f` 是**自由空间波长** $\lambda_0$，所以 `2π/lam0` 就是自由空间波数 $k_0$。代码实际上在做：
$$
k_b = k_0 \sqrt{\varepsilon_{r,b}},\quad k_1 = k_0 \sqrt{\varepsilon_{r,1}}
$$

---

## 2. Lippmann–Schwinger 方程（连续形式）

总场 = 入射场 + 散射体的次生辐射：
$$
\boxed{\;E_z(\mathbf r) = E_z^{\text{inc}}(\mathbf r) + k_b^2\!\int_S G(\mathbf r,\mathbf r')\,\chi(\mathbf r')\,E_z(\mathbf r')\,dS'\;}
$$
积分只在圆柱截面 $S$ 上（外面 $\chi=0$）。2D 自由空间 Green 函数（$e^{j\omega t}$ 约定）：
$$
G(\mathbf r,\mathbf r')=\frac{1}{4j}H_0^{(2)}\!\bigl(k_b|\mathbf r-\mathbf r'|\bigr)
$$
$H_0^{(2)}$ 是第二类零阶 Hankel 函数，描述外向柱面波。

> 这是**非线性**方程（$E_z$ 既在左边又在被积函数里），但**正问题里 $\chi$ 已知**，所以对未知量 $E_z$ 而言它是**线性**的——这正是正问题可以一次解出的原因。
### 2.1 第二类零阶 Hankel 函数 (Hnakel function of the second kind of order zero)

$$H_0^{(2)}(x)=J_0(x)-jY_0(x)$$
式中
- $J_0(x)$ 是第一类零阶贝塞尔函数（Bessel function of the first kind），描述柱面驻波的实部；
- $Y_0(x)$ 是第二类零阶贝塞尔函数（Bessel function of the second kind，也称 Neumann函数），描述柱面波的虚部，在原点 $x=0$ 处有奇异性（趋于负无穷）；
- $j$ 是虚数单位。

### 2.2 第一类零阶贝塞尔函数$J_0(x)$

$J_0(x)$ 是贝塞尔微分方程的一个解，它在原点 $x=0$ 处是解析的且取值有限 $(J_0(0)=1)$。其泰勒级数展开式为
$$J_0(x)=\sum_{m=0}^\infty\frac{(-1)^m}{(m!)^2}\left(\frac{x}{2}\right)^{2m}$$
展开前几项为：$$J_0(x)=1-\frac{x^2}{4}+\frac{x^4}{64}-\frac{x^6}{2304}+\cdots$$
### 第二类零阶贝塞尔函数 $Y_0(x)$

$Y_0(x)$ 也称诺伊曼函数（Neumann function）是贝塞尔微分方程的另一种线性无关解。与$J_0(x)$不同，它在原点 $x=0$ 处存在对数奇异性（当 $x\rightarrow 0$时，函数趋向于 $-\infty$）。其标准级数展开公式为

$$Y_0(x)=\frac{2}{\pi}\left[ \left(\gamma +\ln\frac{x}{2}\right)J_0(x)+\sum_{m=1}^\infty\frac{(-1)^{m+1}H_m}{(m!)^2}\left(\frac{x}{2}\right)^{2m}\right]$$
式中
- $\gamma\approx 0.57721566$ 是欧拉-马斯克诺尼常数（Euler-Mascheroni constant）
- $H_m = 1+\frac{1}{2}+\frac{1}{3}+\cdots+\frac{1}{m}$是第$m$个调和数

---

## 3. MoM 离散化（Richmond 法）：积分方程 → 矩阵方程

### 3.1 三步骤思路

1. 把圆柱所在的方形区域分成 $N$ 个边长 $d$ 的小方格（pulse basis：每格内 $E_z$、$\chi$ 取常数）；
2. 在每格中心点配置（point matching）；
3. 把对每格的积分算出来 → 得到矩阵元。

格子设置：只给落在圆柱内的格子赋 $\chi_n\neq0$，圆柱外 $\chi_n=0$（这些格子最终不进未知量，或进了但 $\chi=0$ 自动无贡献）。

### 3.2 离散方程

对第 $m$ 个格子中心 $\mathbf r_m$：

$$
E_m = E_m^{\text{inc}} + k_b^2\sum_{n=1}^{N}\chi_n E_n\underbrace{\int_{\text{cell}_n}G(\mathbf r_m,\mathbf r')\,dS'}_{I_{mn}}
$$

整理成线性系统 $(\mathbf I-\mathbf D)\mathbf E=\mathbf E^{\text{inc}}$，其中 $D_{mn}=k_b^2\chi_n I_{mn}$。剩下的全部工作就是**把 $I_{mn}$ 算准**——尤其是 $m=n$ 的自项（积分核奇异）。

### 3.3 关键技巧：方格 → 等面积圆（Richmond）

直接对方格积分 $H_0^{(2)}$ 没有闭式解。Richmond 的招：把方格换成**同面积的圆盘**，半径

$$
a=\frac{d}{\sqrt\pi}\qquad(\pi a^2=d^2)
$$

圆盘上 $H_0^{(2)}$ 的积分有闭式（用 Bessel 恒等式 $\int_0^a rH_0^{(2)}(k_b r)\,dr=\frac{a}{k_b}H_1^{(2)}(k_b a)$ 和 Graf 加法定理）。

### 3.4 矩阵元（拿来即用，但务必用 Mie 验证符号）

$$
\boxed{
D_{mn}=
\begin{cases}
-\,\chi_n\,\dfrac{j\pi k_b a}{2}\,J_1(k_b a)\,H_0^{(2)}(k_b\rho_{mn}), & m\neq n\\
-\,\chi_n\Bigl[\dfrac{j\pi k_b a}{2}\,H_1^{(2)}(k_b a)+1\Bigr], & m=n
\end{cases}}
$$

其中 $a=d/\sqrt{\pi}$为等面积圆半径；$J_1$ 一阶 Bessel，$H_1^{(2)}$ 一阶第二类 Hankel；$\rho_{mn}=|\mathbf r_m-\mathbf r_n|$，为两格子间距；$m,n$都是**格子编号**，$n$ 源格子编号($1\cdots N$)；$m$ 观测格子编号($1\cdots N$)：
- **$m$ = 观测格(行)**:"我想知道第 $m$ 个格子处的场是多少"——对应矩阵的**行**。
- **$n$ = 源格(列)**:"第 $n$ 个格子里的散射源,贡献到别处"——对应矩阵的**列**。
每个格子既能当观测点又能当源,所以 $D$ 是 $N\times N$ 方阵。$D_{mn}$ = "第 $n$ 格的源对第 $m$ 格的场的贡献"。
系统矩阵 $A_{mn}=\delta_{mn}-D_{mn}$。

**两点推导说明**（便于你自己核对）：
- **非自项**来自"圆盘上 $H_0^{(2)}(k_b|\mathbf r_m-\mathbf r'|)$ 的积分 $=\frac{2\pi a}{k_b}J_1(k_b a)H_0^{(2)}(k_b\rho_{mn})$"，再乘 $\frac{1}{4j}$（Green 前因子）和 $k_b^2\chi_n$。
- **自项**($m=n$,观测在圆心)：$\int_0^a rH_0^{(2)}(k_br)dr=\frac{a}{k_b}H_1^{(2)}(k_ba)-\frac{2j}{\pi k_b^2}$,前一项乘开给出 $-\chi_n\frac{j\pi k_ba}{2}H_1^{(2)}(k_ba)$,后一项($-\frac{2j}{\pi k_b^2}$,源自 $Y_1$ 原点奇异)乘 $k_b^2\chi_n\cdot\frac1{4j}\cdot2\pi$ 给出 **$-\chi_n$**。两项合起来即 $-\chi_n[\frac{j\pi k_ba}{2}H_1^{(2)}(k_ba)+1]$。**漏掉 $-\chi_n$ 会让强散射反演对不上 Mie(弱散射看不出)。**
- 小宗量展开:$\frac{j\pi k_ba}{2}H_1^{(2)}(k_ba)+1\approx j\frac{k_b^2d^2}{4}$,故 $D_{nn}\approx-j\chi_n\frac{k_b^2d^2}{4}$,是 $O(d^2)\to0$——self 项随格子缩小而消失,物理上合理。

### 3.5 这个公式怎么从 L-S 来的(4 步)

L-S 连续方程:

$$E_z(\mathbf r)=E_z^{\text{inc}}(\mathbf r)+k_b^2\int_S G(\mathbf r,\mathbf r')\chi(\mathbf r')E_z(\mathbf r')\,dS'$$

**第 1 步:离散积分(pulse basis)。** 把圆柱区域 $S$ 切成 $N$ 个小格,每格内 $\chi$、$E_z$ 当常数。积分 = 各格贡献之和:

$$\int_S G\chi E_z\,dS'\approx\sum_{n=1}^{N}\chi_n E_n\underbrace{\int_{\text{cell}_n}G(\mathbf r,\mathbf r')\,dS'}_{\text{只剩 Green 在第 }n\text{ 格上的积分}}$$

**第 2 步:在格心配置(point matching)。** 把方程在每个格心 $\mathbf r_m$ 上写一遍($m=1\dots N$),得到 $N$ 个方程:

$$E_m=E_m^{\text{inc}}+k_b^2\sum_{n=1}^{N}\chi_n E_n\,I_{mn},\qquad I_{mn}=\int_{\text{cell}_n}G(\mathbf r_m,\mathbf r')\,dS'$$

**第 3 步:写成矩阵。** 令 $D_{mn}=k_b^2\chi_n I_{mn}$,移项:

$$E_m-\sum_n D_{mn}E_n=E_m^{\text{inc}}\;\Longrightarrow\;(\mathbf I-\mathbf D)\mathbf E=\mathbf E^{\text{inc}}$$

**第 4 步:把 $I_{mn}$ 算出闭式(Richmond 等面积圆)。** 第 $n$ 格换成半径 $a=d/\sqrt\pi$ 的圆盘:

- **$m\neq n$**(从 $\mathbf r_m$ 看远处的圆盘,间距 $\rho_{mn}$):圆盘上 $H_0^{(2)}$ 的积分有闭式 $\frac{2\pi a}{k_b}J_1(k_ba)H_0^{(2)}(k_b\rho_{mn})$,再乘 Green 前因子 $\frac1{4j}$ 和 $k_b^2\chi_n$:

$$D_{mn}=k_b^2\chi_n\cdot\frac{1}{4j}\cdot\frac{2\pi a}{k_b}J_1(k_ba)H_0^{(2)}(k_b\rho_{mn})=-\chi_n\frac{j\pi k_ba}{2}J_1(k_ba)H_0^{(2)}(k_b\rho_{mn})$$

(用了 $\frac{k_b}{2j}=-\frac{jk_b}{2}$。)

- **$m=n$**(从自己格心看自己的圆盘,$\rho=0$ 奇异):加法定理退化,$J_1(k_ba)H_0^{(2)}(k_b\rho)$ 换成 $H_1^{(2)}(k_ba)$:

$$D_{nn}=-\chi_n\Bigl[\dfrac{j\pi k_b a}{2}\,H_1^{(2)}(k_b a)+1\Bigr]$$

**这就是 §3.4 那个 boxed 公式。** 整个 $\mathbf D$ 矩阵,本质就是 L-S 里那个算子 $k_b^2\int G\chi(\cdot)$ 的离散版——把"连续积分"变成"$N\times N$ 矩阵乘向量"。

### 落到代码

- `a = d/np.sqrt(np.pi)` —— 几何量,不是 `a_n`。
- `rho` —— 你广播算的 `(N,N)` **格子间距**矩阵。
- 前因子 `pref = -(1j*np.pi*k_b*a/2)`,非自项 `jv(1,k_b*a)*hankel2(0,k_b*rho)`,自项 `hankel2(1,k_b*a)`,最后 `* chi[None,:]`(列=源格 $n$)。


**两点推导说明**（便于你自己核对）：
- **非自项**来自"圆盘上 $H_0^{(2)}(k_b|\mathbf r_m-\mathbf r'|)$ 的积分 $=\frac{2\pi a}{k_b}J_1(k_b a)H_0^{(2)}(k_b\rho_{mn})$"，再乘 $\frac{1}{4j}$（Green 前因子）和 $k_b^2\chi_n$。
- **自项**把上式里"圆心在别处看圆盘"换成"圆心在自己处看自己"，加法定理退化，$J_1H_0^{(2)}\to H_1^{(2)}(k_b a)$。
- 小宗量展开可验证 $D_{nn}\approx\chi_n\bigl(1-j\frac{k_b^2 d^2}{4}\bigr)$，是 $O(1)$ 而非 $O(d^2)$——这是 Green 对数奇异被积出来的有限贡献，**别误以为自项可忽略**。

> [!warning] 符号是最容易踩的坑
> 上面是 $e^{j\omega t}$/$H^{(2)}$ 约定下的结果。换约定整体符号会变。**不要纠结于先验地确认符号——直接拿 Mie 解析解兜底**：跑出来若散射场和 Mie 差一个共轭（虚部反号），就是约定不一致，把 $H^{(2)}\leftrightarrow H^{(1)}$ 或指数符号翻一下即可。

---

## 4. 入射场

每个格子中心赋值：

$$
E_m^{\text{inc}}=E_0\,e^{-jk_b x_m}
$$
取 $E_0=1$；$x_m$为 x 轴距离。组成 $N\times1$ 向量 $\mathbf E^{\text{inc}}$。

---

## 5. 解线性系统得总场

$$
(\mathbf I-\mathbf D)\,\mathbf E_{\text{tot}}=\mathbf E^{\text{inc}}
$$

- **F1 起步**：$N$ 小（圆柱几个波长、几百~几千格），直接 `numpy.linalg.solve` 即可，先保证对；
- **F2 再优化**：换 BiCGStab（你笔记里推过），或当背景均匀、格点规则时用 CG-FFT（$\mathbf D$ 的 Green 部分是 Toeplitz）。F1 阶段**先别上 FFT**，先把物理对了。

解出的 $\mathbf E_{\text{tot}}$ 是圆柱内每格的总场。

---

## 6. 算观测点散射场（$G_{tr}$）

在圆柱外取一圈观测点 $\mathbf r_r$（半径 $R_{\text{obs}}>R_{\text{cyl}}$，均匀 $N_{\text{rx}}$ 个角度）。散射场 = 等效源经 Green 辐射出去：

$$
\boxed{\;E_z^{\text{sc}}(\mathbf r_r)=k_b^2\sum_{n=1}^{N}G(\mathbf r_r,\mathbf r_n)\,\chi_n\,E_{\text{tot},n}\,\Delta S\;}
$$

观测点在圆柱外、离每个源格都有距离，$\rho>0$ 不奇异，直接用 $G=\frac{1}{4j}H_0^{(2)}(k_b\rho)$、$\Delta S=d^2$ 求和即可。这就是你笔记里"体素→接收机"的 $\mathbf G_{tr}$ 矩阵作用。

---

## 7. Mie 解析解（真值）

平面波打介质圆柱，TM 极化，所有场都按照角度展开成一圈圈的模式$e^{jn\phi}(n=\cdots, -1, 0, 1,\cdots)$，三套展开（$E_0=1$）：

$$
\begin{aligned}
\text{入射:}\quad & E_z^{\text{inc}}=\sum_{n=-\infty}^{\infty}(-j)^n J_n(k_b\rho)\,e^{jn\phi}\\
\text{散射(圆柱体外):}\quad & E_z^{\text{sc}}=\sum_{n}(-j)^n a_n H_n^{(2)}(k_b\rho)\,e^{jn\phi}\\
\text{圆柱体内部:}\quad & E_z^{\text{int}}=\sum_{n}(-j)^n c_n J_n(k_1\rho)\,e^{jn\phi}
\end{aligned}
$$
这里
- $k_b$ 背景（圆外）波数
- $k_1 = k_0\sqrt{\varepsilon_r}$ 圆内波数（介质里波更短）
- $\rho$ 任意场点到圆心的距离。由于下面求解$a_n$时使用圆柱边缘连续性进行求解，因此$\rho=R_{cyl}$。
- $R$ 圆柱体半径 $R_{cyl}$
- $J_n(x)$ 第一类Bessel，驻波/规则解，原点有限
- $H_n^{(2)}(x)$ 第二类Hankel，外向行波
- $J_n'(x), H_n^{(2)\prime}(x)$ 对自变量的导数。边界条件要 $\partial_\rho$,而 $\partial_\rho J_n(k\rho)=k\,J_n'(k\rho)$——链式法则那个 $k$ 就是公式里 $k_b/k_1$ 的来
- $a_n$ **第 $n$ 模的散射系数**——这一模被散射出去多强
- $c_n$  内部场系数(被消掉,不关心)

**边界条件**（$\rho=R_{\text{cyl}}$，非磁性 $\mu_1=\mu_b$ → $E_z$ 和 $\partial_\rho E_z$ 都连续）：

物理边界:圆周 $\rho=R$ 上,场 $E_z$ 连续、$\partial_\rho E_z$ 连续(非磁性 → 切向 $H_\phi\propto\partial_\rho E_z$ 也连续)。因为各 $e^{jn\phi}$ 模互相正交,**每个 $n$ 单独满足两个方程**:

$$
\begin{aligned}
J_n(k_b R)+a_n H_n^{(2)}(k_b R)&=c_n J_n(k_1 R)\quad&(1)\ E_z\text{ 连续}\\
k_b\bigl[J_n'(k_b R)+a_n H_n^{(2)\prime}(k_b R)\bigr]&=k_1 c_n J_n'(k_1 R)\quad&(2)\ \partial_\rho E_z\text{ 连续}
\end{aligned}
$$

两个方程、两个未知数 $(a_n,c_n)$。**消掉 $c_n$**:由 (1) 得 $c_n=\dfrac{J_n(k_b R)+a_n H_n^{(2)}(k_b R)}{J_n(k_1 R)}$,代入 (2),两边乘 $J_n(k_1R)$:

$$
k_b J_n(k_1R)\bigl[J_n'(k_bR)+a_nH_n^{(2)\prime}(k_bR)\bigr]=k_1 J_n'(k_1R)\bigl[J_n(k_bR)+a_nH_n^{(2)}(k_bR)\bigr]
$$

把含 $a_n$ 的项归到一边:

$$
a_n\bigl[k_bJ_n(k_1R)H_n^{(2)\prime}(k_bR)-k_1J_n'(k_1R)H_n^{(2)}(k_bR)\bigr]=k_1J_n'(k_1R)J_n(k_bR)-k_bJ_n(k_1R)J_n'(k_bR)
$$

两边同乘 $-1$ 整理,就是:

$$
\boxed{\;a_n=-\,\frac{k_1\,J_n'(k_1 R)\,J_n(k_b R)-k_b\,J_n(k_1 R)\,J_n'(k_b R)}{k_1\,J_n'(k_1 R)\,H_n^{(2)}(k_b R)-k_b\,J_n(k_1 R)\,H_n^{(2)\prime}(k_b R)}\;}
$$

**所以 $a_n$ 不是凭空的——它就是"$E_z$ 连续 + 导数连续"这两个边界条件解出来的 2×2 线性方程(本质是 Cramer 法则),分子分母全是这两条边界关系的组合。**
消去 $c_n$ 解出散射系数：

> [!note] 实现细节
> - $J_n,H_n^{(2)}$ 及其导数：`scipy.special` 的 `jv, hankel2, jvp, h2vp`（导数用 `jvp(n,x)`、`h2vp(n,x)`）。
> - 级数截断：$n$ 从 $-N_{\max}$ 到 $+N_{\max}$，取 $N_{\max}\approx k_b R_{\text{cyl}}+10$（经验，物体越大项越多）。验证：增大 $N_{\max}$ 结果不再变即收敛。
> - 在你 MoM 用的同一圈观测点上求 $E_z^{\text{sc,Mie}}$，才能逐点对比。

### 算出 $a_n$ 有什么用、意义何在

**$a_n$ 是第 $n$ 个角向模的散射强度。** 把所有 $a_n$ 塞回散射级数 $E_z^{sc}=\sum(-j)^na_nH_n^{(2)}(k_b\rho)e^{jn\phi}$,就得到**圆外任意位置的精确散射场**——这就是你要的**解析真值**。

意义:

1. **它是你 MoM 的标尺。** F1 整个目的就是"MoM 算的散射场 vs Mie 解析散射场",误差小才说明 MoM 写对了。没有 $a_n$,你就没有独立真值去验证。
2. **物理直觉**:小圆柱只有低阶模 $n=0,\pm1$ 被显著激发;圆柱越大($k_bR$ 越大),被激发的模越多——这正是 $N_{\max}\approx k_bR+10$ 的来历(截断到够用的模数)。
3. 它纯解析、零误差(只要级数收敛),所以拿它当 golden reference 最干净。

落到代码:`mie_an(n,...)` 就是上面那个分式;`mie_scattered` 把 $n=-N_{\max}\dots N_{\max}$ 的 $(-j)^na_nH_n^{(2)}(k_b\rho)e^{jn\phi}$ 累加起来。你现在可以照上一条消息的 4 步写 `mie_scattered` 了,写完贴来我核对。

### 什么是"角向模"?从最基础讲

#### 第一步:圆周上的傅里叶分解

先记住一个数学事实:**任何绕圆一圈的图案(关于角度 $\phi$ 周期 $2\pi$),都能拆成一堆"纯旋转花样" $e^{jn\phi}$ 的叠加**($n=0,\pm1,\pm2,\dots$)。这就是"圆周上的傅里叶级数"。每个整数 $n$ 对应一个"角向模"(也叫角向谐波、分波)。
这里的 $\phi$ **正是你猜的**: 观测点相对 $+x$ 轴**逆时针**转过的角度(`np.arctan2(y, x)`)。

#### 第二步:每个 $n$ 长什么样——看"瓣数"

$e^{jn\phi}$ 绕一圈($\phi:0\to2\pi$)时震荡 $|n|$ 次,$|n|$ 越大,角向花样越细、瓣越多:

- **$n=0$**:$e^{0}=1$,**完全均匀**,各方向一样(像一个均匀的圈,没有方向性)。
- **$n=1$**:$\cos\phi$ 型,绕一圈一上一下 → **2 瓣**(偶极,像"左右"图案)。
- **$n=2$**:绕一圈两上两下 → **4 瓣**(四极)。
- **$n=3$**:**6 瓣**……$|n|$ 越大瓣越密。

我画给你看:
![[Pasted image 20260609170850.png]]
### 第三步:为什么要拆成模——因为它们互不干扰

圆柱是**旋转对称**的。这带来一个魔法:**一个纯 $e^{jn\phi}$ 的入射花样,散射出来还是纯 $e^{jn\phi}$,只是被缩放了一个倍数。不同 $n$ 之间不会互相混。** 于是"一个难的散射问题"被拆成"无穷多个独立的简单问题",每个 $n$ 只需解一个 2×2(就是 $a_n$ 的来历)。这就是用模分解的全部动机。

### 第四步:$a_n$ 到底是什么——纠正一个理解

$a_n$ 是**第 $n$ 个角向花样在散射场里的复权重(含大小和相位)**。

⚠️ 注意:$a_n$ **不是**"某个观测角 $\phi$ 上的散射强度"。某个角 $\phi$ 上的实际场,是**所有模在那个 $\phi$ 上叠加的总和**:

$$E^{sc}(\rho,\phi)=\sum_n \underbrace{(-j)^n a_n}_{\text{第 }n\text{ 模的权重}}\,\underbrace{H_n^{(2)}(k_b\rho)}_{\text{径向}}\,\underbrace{e^{jn\phi}}_{\text{角向花样}}$$

**$a_n$ 是"第 $n$ 个旋转频率"的振幅,场是把所有频率加起来**。完全类比声音的傅里叶分解——把声音拆成纯音(频率),$a_n$ 就是每个纯音的响度。这里把场拆成"旋转频率"$n$。

### 第五步:小圆柱 vs 大圆柱 → $N_{\max}$

- **小圆柱**($k_bR$ 小):只有低阶模 $n=0,\pm1$ 被显著激发,散射图案平滑;
- **大圆柱**:要很多高阶模才能拼出细致的角向结构(尖锐的前向瓣、后面的阴影)。

这正是 $N_{\max}\approx k_bR_{\text{cyl}}+10$ 的来历——截断到"够用的模数"。物体越大,要加的瓣越多。

一句话总结:**角向模 $n$ = 一个绕圈震荡 $|n|$ 次的基本花样($e^{jn\phi}$);$a_n$ = 这个花样在散射场里占多重;总散射场 = 所有花样按 $a_n$ 加权叠加。** $\phi$ 就是你说的那个逆时针观测角。

---

## 8. 对比与收敛曲线

**误差度量**（同一圈观测点上）：

$$
\text{err}=\frac{\|\mathbf E^{\text{sc,MoM}}-\mathbf E^{\text{sc,Mie}}\|_2}{\|\mathbf E^{\text{sc,Mie}}\|_2}
$$

**两张必出的图**：
1. **逐点叠加**：固定网格，画 $\text{Re}(E^{\text{sc}})$、$\text{Im}(E^{\text{sc}})$ 沿观测角的 MoM（点）vs Mie（线）。对得上 = 物理对了。
2. **收敛曲线**：横轴"每波长格数" $N_\lambda=\lambda_1/d$（注意用**介质内**短波长 $\lambda_1=\lambda_0/\sqrt{\varepsilon_r}$），纵轴 err，log-log。应单调下降、斜率约 $1\sim2$。这张图就是 F1 的成果证据。

---

## 9. 实现路线图（函数清单，你来填实现）

建议拆成这几个纯函数（便于单测，也便于将来 $\mathbf A\mathbf v$/$\mathbf A^H\mathbf u$ 复用）：

```python
# --- 几何与网格 ---
def make_grid(domain_size, d):            # -> 格子中心坐标 (N,2), ΔS
def assign_contrast(centers, R_cyl, eps_r, eps_b):  # -> χ (N,) 复数

# --- MoM 正演 ---
def green_2d(k_b, R):                     # (1/4j) H0^(2)(k_b R)，R 可为数组
def build_D(centers, chi, k_b, d):        # -> D (N,N)，含 self-cell 闭式
def incident_plane_wave(centers, k_b, E0=1):   # -> E_inc (N,)
def solve_total_field(D, E_inc):          # 解 (I-D)E = E_inc -> E_tot (N,)
def scattered_field(rx_points, centers, chi, E_tot, k_b, dS):  # -> E_sc (Nrx,)

# --- Mie 解析 ---
def mie_an(n, k_b, k_1, R_cyl):           # 散射系数 a_n
def mie_scattered(rx_points, k_b, k_1, R_cyl, Nmax):  # -> E_sc_mie (Nrx,)

# --- 验证 ---
def rel_l2_error(a, b)
def convergence_study(list_of_d):         # 循环不同 d，画 err vs Nλ
```

**建议参数（第一次跑，弱散射、易对）**：
- 背景真空 $\varepsilon_b=1$，圆柱 $\varepsilon_r=2.0$（弱），半径 $R_{\text{cyl}}=0.5\lambda_0$；
- 频率随意（如 1 GHz），$\lambda_0=c/f$；
- 网格 $d=\lambda_1/15$ 起步，收敛研究再扫 $d=\lambda_1/\{8,10,15,20,30\}$；
- 观测圈 $R_{\text{obs}}=3R_{\text{cyl}}$，$N_{\text{rx}}=72$（每 5°）。

对通了再加难：$\varepsilon_r=4\sim10$（强散射，考验 MoM 而非 Born）、更大圆柱。

---

## 10. 常见坑（按踩中概率排序）

| 坑 | 症状 | 处理 |
|---|---|---|
| **约定不一致** | 散射场和 Mie 差共轭/反号 | Green、入射、Mie 三处统一 $e^{j\omega t}$+$H^{(2)}$；翻一处试 |
| **self-cell 用错** | 误差怎么加密都降不下来、或 NaN | 自项必须用 $H_1^{(2)}(k_b a)$ 闭式，不能跳过或当 0 |
| **等效半径写错** | 误差偏大且不收敛 | $a=d/\sqrt\pi$（等面积），不是 $d/2$ |
| **波长用错** | 收敛曲线横轴尺度怪 | 网格密度按**介质内** $\lambda_1=\lambda_0/\sqrt{\varepsilon_r}$ 算，内部波长更短 |
| **Mie 截断太小** | Mie 自己就不准 | $N_{\max}\approx k_bR+10$，加大到结果不变 |
| **观测点落进圆柱** | 散射场发散 | $R_{\text{obs}}>R_{\text{cyl}}$ |
| **只赋圆内格子但矩阵含圆外** | 维度/索引乱 | 要么只保留 $\chi\neq0$ 的格子进未知量，要么全保留但圆外 $\chi=0$（更简单，先用这个） |
| **$\Delta S$ 漏乘** | 散射场幅度系统性偏差 | 第 6 节求和别忘 $\Delta S=d^2$ |

---

## 11. 自测清单（里程碑式，逐项打勾）

- [ ] **T1 几何**：画出网格 + 圆柱边界 + 观测圈，目视无误。
- [ ] **T2 入射场**：$|E^{\text{inc}}|\equiv1$，相位沿 $x$ 线性，等相位面垂直 $x$ 轴。
- [ ] **T3 Green 对称**：$D_{mn}=D_{nm}$ 当 $\chi_m=\chi_n$（互易）。
- [ ] **T4 弱散射 sanity**：$\varepsilon_r=1.01$ 时，MoM 散射场 ≈ Born 单步（$\mathbf E_{\text{tot}}\approx\mathbf E^{\text{inc}}$）。
- [ ] **T5 Mie 自收敛**：加大 $N_{\max}$，Mie 散射场稳定不变。
- [ ] **T6 逐点对比**：固定网格，Re/Im 叠加图 MoM 贴合 Mie。
- [ ] **T7 收敛阶**：err vs $N_\lambda$ 单调降，log-log 斜率 $1\sim2$。
- [ ] **T8 强散射**：$\varepsilon_r=8$ 仍能在细网格下 err$<5\%$（证明不是靠弱散射蒙对）。

全部打勾 → **F1 通过**，可以进 F2（CG-FFT 加速）。

---

## 12. 参考

- Richmond, J.H. (1965). *Scattering by a dielectric cylinder of arbitrary cross section shape.* IEEE Trans. AP-13. —— MoM 矩阵元与等效圆 self-cell 的原始出处。
- Harrington, *Field Computation by Moment Methods.* —— MoM 通用框架。
- Balanis / Bohren & Huffman —— 圆柱 Mie 级数与边界条件标准推导。
- 你自己的笔记：`[[BIM 正向求解器]]` 第 1–2 章（Green/MoM 离散）、`[[第6章 CG]]`、第 7 章 CG-FFT（F2 用）。

> [!tip] 和硬件线的接口预埋
> 把第 5、6 节写成"算子"形式（给 $\mathbf v$ 返回 $\mathbf A\mathbf v$），F2 换 FFT 时只动算子内部、不动外面。这个 FFT 算子将来正好对应 Zenith-Radar 的 1D/2D-FFT 核——两个项目从这里开始对齐。
