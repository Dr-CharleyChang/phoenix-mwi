# P1 Tutorial: from separate algorithms to one evaluated imaging platform

> P1-A = unified evaluation, P1-B = reproducible Born/DBIM/CSI benchmark and report, P1-C = DAS qualitative imaging. This tutorial starts with school-level vector arithmetic, derives every metric and the DAS back-projection, then maps the mathematics to the Python files and the automatic driver.

## 0. What has been built

Before P1, Phoenix already had a validated forward solver and three quantitative inverse methods, but each example lived in its own driver. That proved the individual algorithms; it did not yet prove that the repository was a platform. P1 connects the pieces into one controlled experiment in which every method sees the same target, frequency, transmit views, receiver array, and measured data.

~~~mermaid
flowchart LR
    A["Known synthetic contrast chi_true"] --> B["Full nonlinear MoM forward model"]
    B --> C["One shared receiver-data vector d"]
    C --> D["DAS qualitative imager"]
    C --> E["Born inverter"]
    C --> F["DBIM inverter"]
    C --> G["CSI inverter"]
    D --> H["Qualitative metrics"]
    E --> I["Quantitative metrics"]
    F --> I
    G --> I
    H --> J["PNG + JSON + Markdown report"]
    I --> J
~~~

The implementation adds these user-facing pieces:

- **P1-A:** mwisim/evaluation/image_metrics.py provides RMSE, relative permittivity RMSE, SSIM, support IoU, localization error, contrast recovery, full-data residual, and ImageMetricsEvaluator.

- **P1-B:** mwisim/evaluation/benchmark.py runs the common experiment; mwisim/reporting/report.py writes figures, a machine-readable JSON scorecard, and a Markdown report; scripts/run_phase1_benchmark.py is the one-command driver.

- **P1-C:** mwisim/imaging/das.py implements frequency-domain Delay-and-Sum, which in this plane-wave experiment is the coherent Born back-projection.

- **Tests:** tests/test_p1a_evaluation.py, tests/test_p1b_benchmark.py, and tests/test_p1c_das.py make the three milestones executable specifications.

## 1. The minimum mathematics we need

### 1.1 A number, a vector, and a matrix

A scalar is one number, for example the contrast of one cell $\chi_3=0.5$. A vector is an ordered list of numbers, for example the contrast values of four cells:

$$
\boldsymbol\chi=
\begin{bmatrix}
0\\
0.5\\
0.5\\
0
\end{bmatrix}.
$$

A matrix is a rectangular table of numbers. Multiplying a matrix by a vector forms weighted sums. If

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
5\\
6
\end{bmatrix},
$$

then

$$
Ax=
\begin{bmatrix}
1\cdot5+2\cdot6\\
3\cdot5+4\cdot6
\end{bmatrix}
=
\begin{bmatrix}
17\\
39
\end{bmatrix}.
$$

In Phoenix, the image is stored as a flat vector because linear algebra works naturally with vectors. For display only, a vector with $N=N_yN_x$ values is reshaped into an $N_y\times N_x$ image. Reshaping changes the arrangement, not the values.

### 1.2 Complex numbers and conjugation

A frequency-domain electric field is generally complex:

$$
E=a+jb.
$$

Its conjugate reverses the sign of the imaginary part:

$$
\overline E=E^*=a-jb.
$$

Multiplying a phasor by its conjugate removes its phase and produces its squared magnitude:

$$
E^*E=|E|^2=a^2+b^2.
$$

This is why back-projection uses conjugated propagation terms. Forward propagation adds a phase delay; conjugated propagation unwinds that delay so contributions from the correct candidate cell line up coherently.

### 1.3 Contrast and relative permittivity

Phoenix reconstructs contrast rather than relative permittivity directly:

$$
\chi=\frac{\varepsilon_r}{\varepsilon_b}-1.
$$

Solving this equation for relative permittivity gives

$$
\varepsilon_r=\varepsilon_b(1+\chi).
$$

For the Phase-1 vacuum background $\varepsilon_b=1$, a true target with $\varepsilon_r=1.5$ has $\chi=0.5$. If an algorithm estimates $\hat\chi=0.4$, it implies $\hat\varepsilon_r=1.4$.

### 1.4 Dimensions of the common Phase-1 problem

Let $N$ be the number of imaging cells, $M$ the number of receivers, and $N_v$ the number of plane-wave views.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\chi$ | $(N,)$ | contrast map |
| centers | $(N,2)$ | $(x,y)$ coordinate of every cell |
| rx | $(M,2)$ | receiver coordinates |
| $E_i^{\mathrm{inc}}$ | $(N,)$ | incident field in all cells for view $i$ |
| E_inc_set | $(N_v,N)$ | all incident fields |
| $G_{\mathrm{tr}}$ | $(M,N)$ | propagation from each cell to each receiver |
| $d_i$ | $(M,)$ | receiver data for view $i$ |
| $d$ | $(N_vM,)$ | all view blocks stacked into one vector |
| $\hat\chi$ | $(N,)$ | quantitative reconstruction |
| DAS image | $(N,)$ | qualitative normalized intensity map |

The stacking order is view by view:

$$
d=
\begin{bmatrix}
d_1\\
d_2\\
\vdots\\
d_{N_v}
\end{bmatrix}.
$$

Therefore Python can recover the two-dimensional data table with d.reshape(Nv, M). Row $i$ then contains the $M$ receiver samples of view $i$.

## 2. Why all methods must use one shared problem

Suppose Born is tested with a weak target, DBIM with a strong target, and CSI with a different number of receivers. Their errors cannot be compared because both the algorithm and the experiment changed. A fair benchmark changes only the method.

P1 first creates one contrast map $\chi_{\mathrm{true}}$, then generates measurements with the full nonlinear MoM model:

$$
(I-D(\chi_{\mathrm{true}}))E_i=E_i^{\mathrm{inc}},
$$

$$
d_i=S\left(\chi_{\mathrm{true}}\odot E_i\right).
$$

The same stacked $d$ is passed unchanged to DAS, Born, DBIM, and CSI. Born is therefore allowed to suffer from its weak-scattering approximation; that model error is part of the honest comparison.

The word physical in make_born_problem(mode="physical") does not mean measured clinical data. It means that synthetic data came from the full multiple-scattering forward equation rather than from the same linear Born operator used for inversion.

## 3. P1-A — evaluation from arithmetic to scientific meaning

No single number answers every reconstruction question. We want to know at least five different things:

1. Are the cell values numerically close to truth?

2. Is the object shape structurally similar?

3. Is the object in the right physical location?

4. Is the detected support the right size and shape?

5. If the estimate is inserted into the full forward model, does it reproduce the measured signals?

### 3.1 Error vector

For truth $\chi$ and estimate $\hat\chi$, the cellwise error is

$$
e=\hat\chi-\chi.
$$

Take the four-cell example

$$
\chi=[0,\ 1,\ 1,\ 0],
\qquad
\hat\chi=[0,\ 0.8,\ 1.2,\ 0].
$$

Then

$$
e=[0,\ -0.2,\ 0.2,\ 0].
$$

The second cell is underestimated by $0.2$, while the third is overestimated by $0.2$.

### 3.2 RMSE

The root-mean-square error squares every error, averages the squares, and takes a square root:

$$
\operatorname{RMSE}=
\sqrt{\frac{1}{N}\sum_{n=1}^{N}|\hat\chi_n-\chi_n|^2}.
$$

For the four-cell example:

$$
\operatorname{RMSE}
=\sqrt{\frac{0^2+(-0.2)^2+(0.2)^2+0^2}{4}}
=\sqrt{0.02}
\approx0.1414.
$$

Why square first? Positive and negative errors must not cancel. Why take the square root at the end? It restores the original unit of the variable.

Lower RMSE is better, and zero is perfect. RMSE is absolute: an error of $0.1$ means the same numerical difference regardless of how large the truth is.

### 3.3 Relative $L_2$ error

The Euclidean length of a vector is

$$
\|x\|_2=\sqrt{\sum_n|x_n|^2}.
$$

Relative $L_2$ error compares the length of the error with the length of the truth:

$$
e_{\mathrm{rel}}=\frac{\|\hat\chi-\chi\|_2}{\|\chi\|_2}.
$$

For the same example:

$$
\|\hat\chi-\chi\|_2=\sqrt{0.08},
\qquad
\|\chi\|_2=\sqrt{2},
$$

so

$$
e_{\mathrm{rel}}=\frac{\sqrt{0.08}}{\sqrt2}=0.2=20\%.
$$

Lower is better. Unlike RMSE, relative $L_2$ is dimensionless and tells us the error relative to the total strength of the true image.

### 3.4 Relative-permittivity RMSE

The clinically meaningful material variable is often $\varepsilon_r$, so the evaluator converts both maps before computing RMSE:

$$
\operatorname{RMSE}_{\varepsilon_r}
=\operatorname{RMSE}\left(\varepsilon_b(1+\hat\chi),\ \varepsilon_b(1+\chi)\right).
$$

For $\varepsilon_b=1$, adding 1 to both maps does not change their difference, so $\operatorname{RMSE}_{\varepsilon_r}=\operatorname{RMSE}_{\chi}$. The explicit conversion is still important because this equality stops being numerically identical when a non-unit background is used.

### 3.5 SSIM from mean, variance, and covariance

RMSE compares cell values independently. Two images can have similar RMSE but very different shapes, so imaging research also uses the Structural Similarity Index, SSIM.

At school level, the mean describes average brightness:

$$
\mu_x=\frac{1}{K}\sum_{k=1}^{K}x_k.
$$

Variance describes how strongly values spread around their mean:

$$
\sigma_x^2=\frac{1}{K}\sum_{k=1}^{K}(x_k-\mu_x)^2.
$$

Covariance describes whether two images rise and fall together:

$$
\sigma_{xy}=\frac{1}{K}\sum_{k=1}^{K}(x_k-\mu_x)(y_k-\mu_y).
$$

SSIM combines local mean agreement, local contrast agreement, and local structural agreement:

$$
\operatorname{SSIM}(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
$$

$C_1$ and $C_2$ are small positive stabilizers that prevent division by zero in dark or constant regions. Phoenix computes these statistics with Gaussian local windows and averages the resulting SSIM map. This is closer to how human vision judges local structure than one whole-image mean.

SSIM equals $1$ for identical images. Higher is better. It can be negative for severely anticorrelated structures, although ordinary reconstruction results are usually between $0$ and $1$.

### 3.6 Support and Intersection-over-Union

The support means the cells considered to contain the object. Phoenix makes a binary mask by retaining cells whose magnitude is at least a chosen fraction of the map maximum. With the default threshold $0.5$:

$$
\mathcal M(x)_n=
\begin{cases}
1,&|x_n|\ge0.5\max_j|x_j|,\\
0,&\text{otherwise}.
\end{cases}
$$

Suppose the true support is cells $\{2,3\}$ and the estimated support is cells $\{3,4\}$. Their intersection contains one cell, $\{3\}$, and their union contains three cells, $\{2,3,4\}$. Therefore

$$
\operatorname{IoU}=\frac{|\text{intersection}|}{|\text{union}|}=\frac13.
$$

IoU is $1$ for exactly matching supports and $0$ when they do not overlap. Higher is better. Because IoU depends on the threshold, every paper or report must state the threshold.

### 3.7 Localization error

Using only the maximum cell is unstable for a homogeneous object because many true cells have exactly the same value. Phoenix therefore compares weighted centroids of the strong regions.

For cell coordinates $r_n=(x_n,y_n)$ and nonnegative weights $w_n$, the centroid is

$$
r_c=\frac{\sum_n w_nr_n}{\sum_n w_n}.
$$

Only cells above the support threshold receive nonzero weights. The localization error is

$$
e_{\mathrm{loc}}=\|\hat r_c-r_c^{\mathrm{true}}\|_2.
$$

For a one-dimensional example, let the true energy be at coordinate $x=1$ and the estimated energy at $x=2$. Their centroids are 1 and 2, so the localization error is $|2-1|=1$ metre. Phoenix reports metres internally and converts to millimetres in the human report.

Lower localization error is better. This metric asks where the object is, not whether its recovered material value is correct.

### 3.8 Contrast recovery

Let $\Omega_{\mathrm{true}}$ be the thresholded true-object support. Phoenix averages the estimated and true real contrast over that same region:

$$
R_{\chi}=
\frac{\operatorname{mean}_{n\in\Omega_{\mathrm{true}}}\operatorname{Re}(\hat\chi_n)}
{\operatorname{mean}_{n\in\Omega_{\mathrm{true}}}\operatorname{Re}(\chi_n)}.
$$

$R_\chi=1$ means 100% contrast recovery, $R_\chi=0.8$ means the mean contrast is 20% too low, and $R_\chi=1.2$ means it is 20% too high. The code also reports $|R_\chi-1|$ as contrast_rel_error.

This metric deliberately uses the true support in a synthetic benchmark. With real clinical data, the true support is unavailable and must come from a reference segmentation or another validated annotation.

### 3.9 Full nonlinear data residual

An image can look plausible while failing to explain the measured electromagnetic data. Phoenix therefore places every reconstructed $\hat\chi$ back into the full nonlinear forward solver:

$$
\hat d=F(\hat\chi).
$$

It then computes

$$
r_d=\frac{\|F(\hat\chi)-d\|_2}{\|d\|_2}.
$$

This is the common, fair signal-space score for Born, DBIM, and CSI. Lower is better. A small residual does not by itself prove that $\hat\chi$ is correct because inverse problems can be non-unique or ill-conditioned; that is why image-space and data-space metrics are both reported.

### 3.10 Metric direction at a glance

| Metric | Best value | Better direction | Main question |
| --- | ---: | --- | --- |
| relative $L_2$ | 0 | lower | Are all contrast values close overall? |
| RMSE / $\varepsilon_r$ RMSE | 0 | lower | What is the typical cell error? |
| SSIM | 1 | higher | Is the spatial structure similar? |
| localization error | 0 m | lower | Is the object in the correct place? |
| support IoU | 1 | higher | Does the detected region overlap the truth? |
| contrast recovery | 1 or 100% | closer to 1 | Is mean material contrast recovered? |
| full data residual | 0 | lower | Does the reconstructed model reproduce the signals? |
| runtime | 0 s | lower, after accuracy is acceptable | How much computation was required? |

## 4. P1-C — DAS from delay-and-sum to $A^Hd$

### 4.1 The time-domain idea

Imagine one pulse emitted toward a hidden reflector. Receiver 1 sees the echo after $\tau_1$, receiver 2 after $\tau_2$, and receiver 3 after $\tau_3$. If a trial point is the true reflector position, geometry predicts those three delays correctly. We shift each receiver trace backward by its predicted delay and add them. The three echoes align and make a large sum.

If the trial point is wrong, the predicted shifts are wrong. Positive and negative waveform parts do not align, so they partly cancel. Repeating this calculation for every trial point creates an intensity image. This is Delay-and-Sum.

### 4.2 Delay becomes phase in a single-frequency experiment

A sinusoid delayed by $\tau$ gains phase:

$$
e^{j\omega(t-\tau)}=e^{j\omega t}e^{-j\omega\tau}.
$$

Since distance $R=c\tau$ and $k=\omega/c$:

$$
e^{-j\omega\tau}=e^{-jkR}.
$$

In frequency-domain DAS, shifting a signal backward is therefore implemented by multiplying by the conjugate phase $e^{+jkR}$. The Green function contains the physically correct cylindrical spreading and phase, so Phoenix uses $G^*$ instead of a bare exponential.

### 4.3 The predicted signature of one cell

For view $i$ and candidate cell $n$, the known incident field at that cell is $E_{i,n}^{\mathrm{inc}}$. Propagation from that cell to receiver $m$ is $G_{\mathrm{tr},mn}$. Under the Born model, the contribution of cell $n$ to one measurement is proportional to

$$
a_{im,n}=k_b^2dS\,G_{\mathrm{tr},mn}E_{i,n}^{\mathrm{inc}}.
$$

Fixing $n$ and listing $a_{im,n}$ for every view and receiver gives the complete predicted measurement signature of a unit scatterer at cell $n$. In matrix language, that signature is column $n$ of the Born matrix $A$.

### 4.4 Forward projection

The Born forward model adds the signatures of all cells, weighted by their contrasts:

$$
d_{im}\approx k_b^2dS\sum_{n=1}^{N}G_{\mathrm{tr},mn}E_{i,n}^{\mathrm{inc}}\chi_n.
$$

This maps an image $\chi$ to predicted receiver data $d$, so $A\chi$ can be called a forward projection.

### 4.5 Back-projection

To test candidate cell $n$, compare the observed data with that cell's predicted signature. For complex vectors, correlation uses a conjugate:

$$
b_n=\sum_{i=1}^{N_v}\sum_{m=1}^{M}a_{im,n}^*d_{im}.
$$

Substituting $a_{im,n}$ gives

$$
b_n=(k_b^2dS)^*\sum_i\left(E_{i,n}^{\mathrm{inc}}\right)^*\sum_mG_{\mathrm{tr},mn}^*d_{im}.
$$

The vector of all $b_n$ is

$$
b=A^Hd.
$$

This is why DAS is a back-projection and why the P1-C unit test compares coherent_backprojection with BornOperator.rmatvec. The DAS function omits the common scalar $(k_b^2dS)^*$ because the final image is normalized; multiplying every cell by the same nonzero number cannot change the normalized spatial pattern.

### 4.6 A two-cell example with only ordinary arithmetic

Suppose there are two possible target cells and two measurements. Their predicted signatures are the columns of

$$
A=
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}.
$$

Cell 1 would produce signature $[1,1]^T$; cell 2 would produce $[1,-1]^T$. Let the true object occupy only cell 1:

$$
\chi=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

The forward data are

$$
d=A\chi=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

Back-project:

$$
A^Hd=
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
2\\
0
\end{bmatrix}.
$$

Cell 1 receives two aligned contributions, $1+1=2$. Cell 2 receives cancellation, $1-1=0$. DAS therefore peaks at the correct cell without solving an inverse system.

### 4.7 Sensitivity correction

Some cells naturally couple more strongly to the array because they are closer to receivers or more strongly illuminated. A raw back-projection may favor those cells even when they do not contain the target.

The squared norm of candidate cell $n$'s signature, ignoring the common scalar, is

$$
s_n^2=\sum_i|E_{i,n}^{\mathrm{inc}}|^2\sum_m|G_{\mathrm{tr},mn}|^2.
$$

Phoenix divides the coherent sum by $s_n$:

$$
\tilde b_n=\frac{b_n}{\max(s_n,\epsilon)}.
$$

This is column normalization of the matched filter. The tiny positive $\epsilon$ prevents division by zero.

### 4.8 From complex back-projection to a display image

The complex coherent map still has phase. DAS displays nonnegative intensity:

$$
I_n=|\tilde b_n|^p.
$$

Phoenix uses $p=2$ by default, which resembles power or energy. It then divides by the maximum:

$$
I_n^{\mathrm{norm}}=\frac{I_n}{\max_jI_j}.
$$

The result lies between 0 and 1.

### 4.9 What DAS does and does not provide

DAS answers “where is scattering energy likely to be?” It is fast, robust, and useful as an immediate preview or initializer. It does not solve for calibrated material contrast, does not model multiple scattering during imaging, and does not directly return $\varepsilon_r$. Therefore the benchmark reports SSIM, localization, IoU, and runtime for DAS, but intentionally does not report contrast RMSE or contrast recovery for it.

Born, DBIM, and CSI answer a harder quantitative question: “what contrast values best explain the data under this model and regularization?” That is why DAS belongs behind Imager while the other three belong behind Inverter.

## 5. P1-B — one reproducible benchmark

### 5.1 Fair-comparison rules

The benchmark enforces these rules:

1. Generate one full-wave physical synthetic data set.

2. Pass the same arrays to all methods.

3. Use one evaluator implementation and one support threshold.

4. Re-simulate every quantitative result with the same full nonlinear forward model.

5. Measure wall-clock runtime with time.perf_counter.

6. Save both arrays/metrics for machines and figures/text for humans.

7. Record algorithm-specific histories without pretending that differently defined residuals are identical.

### 5.2 Why Born is a shared warm start

Born is very fast and supplies a first approximate contrast map. DBIM and CSI can start from zero, but starting from the Born estimate is a common engineering workflow:

$$
\hat\chi_{\mathrm{Born}}
\longrightarrow
\begin{cases}
\text{DBIM nonlinear refinement},\\
\text{CSI nonlinear refinement}.
\end{cases}
$$

The benchmark computes Born once. DBIM and CSI receive a copy as x0. Their reported total runtime equals

$$
t_{\mathrm{total}}=t_{\mathrm{Born}}+t_{\mathrm{nonlinear\ refinement}}.
$$

The report also lists refinement_runtime_s so we can distinguish initialization cost from nonlinear work. This prevents the shared Born stage from disappearing from timing.

### 5.3 End-to-end pseudocode

~~~text
build one physical synthetic problem
    chi_true, centers, receiver geometry, incident fields, measured d

make one ImageMetricsEvaluator

start timer
chi_born = Born(d)
stop timer
score chi_born in image space
run full forward F(chi_born)
score common full-data residual

start timer
chi_dbim = DBIM(d, x0=chi_born)
stop timer
score chi_dbim in image space
run full forward F(chi_dbim)
score common full-data residual

start timer
chi_csi = CSI(d, x0=chi_born)
stop timer
score chi_csi in image space
run full forward F(chi_csi)
score common full-data residual

start timer
das_map = DAS(d)
stop timer
score location, support, and structure

evaluate acceptance gates
return one structured result dictionary

reporter(result)
    save method-comparison PNG
    save residual-comparison PNG
    save metric-only JSON
    save Markdown report
~~~

### 5.4 The important CSI residual distinction

DBIM stores

$$
\frac{\|d-F(\chi_k)\|}{\|d\|},
$$

which is a full nonlinear forward residual.

CSI internally treats the contrast sources $W$ as optimization variables and stores

$$
\frac{\|d-SW_k\|}{\|d\|}.
$$

This is a valid CSI diagnostic, but $W_k$ may not yet equal $\chi_k\odot E_k$. Therefore it is not automatically equal to $\|d-F(\chi_k)\|/\|d\|$. The report's left residual panel re-simulates all final contrast maps and is the fair cross-method comparison. The right panel shows each method's internal history and labels the definitions explicitly.

### 5.5 Acceptance gates

The automatic run checks:

- DBIM's common full-forward residual is lower than Born's.

- CSI's common full-forward residual is lower than Born's.

- The DAS strong-region centroid lies inside the true object radius.

- Every returned image contains finite numbers.

These gates test integration behavior. They do not claim that DBIM must always beat CSI, or that one fixed parameter set is optimal for every target.

## 6. How the Python implementation follows the mathematics

### 6.1 mwisim/evaluation/image_metrics.py

_component selects real, imaginary, or magnitude data. _image validates or infers the two-dimensional shape. These private helpers prevent every metric from silently reshaping data differently.

rmse computes $\sqrt{\operatorname{mean}(|\hat x-x|^2)}$. eps_r_rmse first applies $\varepsilon_r=\varepsilon_b(1+\chi)$. ssim_2d computes Gaussian local means, variances, covariance, and the standard SSIM expression.

support_mask creates the thresholded Boolean array. support_iou counts logical AND and logical OR cells. energy_centroid calculates a coordinate-weighted average. localization_error subtracts estimated and true centroids. contrast_recovery_ratio averages real contrast inside the true support.

ImageMetricsEvaluator collects these helpers behind the common Evaluator.score interface and is registered as evaluator/image_metrics.

### 6.2 mwisim/imaging/das.py

_validated_arrays checks all required dictionary keys and dimensions before doing physics. This is important because a wrong stacking shape can produce a plausible but meaningless image.

coherent_backprojection reshapes $d$ into $(N_v,M)$, computes the receiver back-propagation for all views, multiplies by the conjugate incident field, sums over views, and optionally divides by column sensitivity.

The central vectorized statements correspond to

$$
R_{i,n}=\sum_m d_{i,m}G_{\mathrm{tr},mn}^*,
$$

and

$$
b_n=\sum_i(E_{i,n}^{\mathrm{inc}})^*R_{i,n}.
$$

das_intensity computes $|b|^p$ and normalizes it. DASImager exposes this through the Imager interface and is registered as imager/das.

### 6.3 mwisim/evaluation/benchmark.py

grid_shape verifies that centers form a complete regular grid. full_forward_prediction calls the nonlinear DBIM forward helper so every method is judged by identical physics.

_timed_reconstruction handles the stopwatch. _method_record runs one inverter, computes image metrics, computes the full nonlinear data residual, and returns a consistent record.

run_phase1_benchmark builds the problem when none is supplied, creates default algorithms, computes Born, uses it as the optional warm start, runs DBIM/CSI/DAS, evaluates gates, and returns one result dictionary. Arrays remain in this in-memory dictionary so the reporter can draw them.

### 6.4 mwisim/reporting/report.py

BenchmarkReporter.write creates four artifacts:

| Artifact | Purpose |
| --- | --- |
| fig_phase1_methods.png | true contrast, DAS, Born, DBIM, and CSI in one row |
| fig_phase1_residuals.png | common final residuals plus clearly labelled internal histories |
| phase1_benchmark_metrics.json | machine-readable parameters, metrics, timings, and acceptance gates |
| phase1_benchmark_report.md | human-readable experiment report with tables and embedded figures |

The JSON intentionally excludes large estimate arrays. A benchmark scorecard should stay small enough for version control and later statistical aggregation. The PNGs carry visual maps; the live Python result retains arrays.

### 6.5 scripts/run_phase1_benchmark.py

The driver contains orchestration, not physics. main performs only three jobs: call run_phase1_benchmark, pass the result to BenchmarkReporter, and print a compact terminal scorecard.

This separation matters. A notebook, command-line tool, web service, or future YAML pipeline can call the same library functions without copying algorithm code from the script.

### 6.6 Why the CSI implementation became faster

For fixed $\chi$, CSI solves the same least-squares matrix $B$ for every transmit view:

$$
Bw_i\approx b_i.
$$

The original code called numpy.linalg.lstsq once per view. Linear algebra libraries can solve several right-hand sides at once:

$$
B
\begin{bmatrix}
w_1&w_2&\cdots&w_{N_v}
\end{bmatrix}
\approx
\begin{bmatrix}
b_1&b_2&\cdots&b_{N_v}
\end{bmatrix}.
$$

The optimized update stacks all $b_i$ as columns and calls numpy.linalg.lstsq once. It computes the same mathematical solution while reusing the expensive matrix factorization.

## 7. Tests as executable specifications

### 7.1 P1-A tests

- P1A.1 checks the four-cell RMSE arithmetic and contrast recovery by hand-computable values.

- P1A.2 checks that identical images produce SSIM $=1$ and IoU $=1$.

- P1A.3 shifts a one-cell object by exactly one metre and checks localization error $=1$ and IoU $=0$.

- P1A.4 checks the full evaluator scorecard and registry construction.

### 7.2 P1-C tests

- P1C.1 proves numerically that DAS coherent back-projection equals Born $A^Hd$ after restoring the common $(k_b^2dS)^*$ scalar.

- P1C.2 uses full nonlinear synthetic data and checks finite normalized output, localization inside the cylinder, and nonzero support overlap.

- P1C.3 checks the Imager registry and display reshaping.

### 7.3 P1-B tests

- P1B.1 checks that one run returns DAS plus all three quantitative methods and every expected metric.

- P1B.2 checks that DBIM and CSI improve the common nonlinear data fit over Born and that DAS/all-finite gates pass.

- P1B.3 checks that warm-start timing includes the shared Born cost.

- P1B.4 writes a report into pytest's temporary directory, verifies all files are nonempty, parses the JSON, and confirms the Markdown table contains all methods.

### 7.4 Testing pyramid

Most new tests are small arithmetic or shape tests and finish quickly. One module-scoped integration fixture runs the complete small benchmark once and shares its result among four tests. The older I1/I2/I3 tests remain algorithm-specific scientific gates. The full suite then checks that P1 integration did not break F1/F2 forward validation or Phase-0 interfaces.

## 8. Run it yourself

From C:\Projects\Project_Pheonix\mwi:

    python -m pytest tests/test_p1a_evaluation.py tests/test_p1b_benchmark.py tests/test_p1c_das.py -q

Run the complete repository:

    python -m pytest -q

Generate the real report artifacts:

    python scripts/run_phase1_benchmark.py

The default benchmark is deliberately small enough for frequent CPU execution: $9\times9=81$ cells, 8 plane-wave views, 20 receivers, $1$ GHz, and $\varepsilon_r=1.5$. It is not intended to be the final publication-resolution experiment.

The verified reference run produced approximately:

| Method | $\chi$ relative $L_2$ | SSIM | Common full-data residual |
| --- | ---: | ---: | ---: |
| Born | 0.6146 | 0.7100 | $3.5004\times10^{-1}$ |
| DBIM | 0.3792 | 0.8536 | $1.6231\times10^{-3}$ |
| CSI | 0.4207 | 0.7788 | $1.1436\times10^{-1}$ |

Exact runtime varies by computer, BLAS library, CPU load, and Python environment. Small floating-point differences in reconstruction scores are also normal.

## 9. How to interpret the generated pictures

In fig_phase1_methods.png, the true panel is the target. DAS should show one central energy blob but need not reproduce the flat target value. Born typically locates the object but spreads and underestimates/overestimates contrast because it ignores multiple scattering. DBIM repeatedly updates both field and Jacobian, so it should sharpen the quantitative map and strongly reduce full-data residual. CSI jointly adjusts contrast sources and contrast; with these simple unoptimized hyperparameters it improves over Born but need not beat DBIM.

In fig_phase1_residuals.png, compare method heights only in the left panel because those bars share exactly the same definition. Use the right panel to judge whether each nonlinear algorithm progresses across its own iterations.

## 10. Common mistakes and troubleshooting

### “DAS is another inverse solver, so why not report its $\chi$ RMSE?”

DAS outputs normalized intensity, not calibrated contrast. A value 1 means “strongest candidate cell,” not $\chi=1$. Reporting $\chi$ RMSE would compare unlike physical quantities.

### “The smallest data residual must be the best image.”

Not necessarily. An ill-posed inverse problem can fit noise or exploit weakly observable directions. Check image metrics, data residual, regularization, and eventually performance on independent/noisy or measured data.

### “CSI's internal residual is 6%, so why does the common residual differ?”

CSI's internal residual uses its independent contrast-source variable $W$. The common score throws $W$ away, inserts $\hat\chi$ into the full forward solver, and asks what that physical model predicts.

### “Why does SSIM change when I change the image shape?”

SSIM is spatial. Reshaping with the wrong $(N_y,N_x)$ changes which cells are neighbors and therefore changes local means and variances. grid_shape validates Phoenix's regular-grid layout.

### “Why does IoU change when I change 0.5?”

The threshold defines what counts as detected support. Threshold sensitivity is real, so the value must be fixed in a benchmark and disclosed. Future studies can plot IoU across several thresholds.

### “Why can contrast recovery look good while relative $L_2$ is poor?”

Contrast recovery averages only inside the true support. Positive and negative cell errors can partly cancel in that average, while relative $L_2$ counts every squared error, including background artifacts.

### “Why can the benchmark be fast even though I2/I3 felt slow?”

The default integration grid is intentionally small, and CSI now solves all view right-hand sides together. Publication-quality spatial convergence studies must use larger grids and report the increased runtime.

## 11. Scientific scope and what P1 does not prove

P1 proves that Phoenix has a reproducible 2-D synthetic vertical slice and that its interfaces connect correctly. It does not prove breast-cancer diagnosis, bone-density accuracy, patient safety, or clinical generalization.

The present benchmark has one homogeneous centered cylinder, one frequency, ideal plane waves, a full circular receiver array, exact geometry, no calibration error, no antenna coupling, no skin layer, no tissue dispersion, no noise model, and no measured-data uncertainty. These simplifications are useful for software verification because the cause of a failure is controllable, but they make the problem much easier than realistic MWI.

## 12. What comes after P1-A/B/C

The next work should harden this vertical slice before adding more inversion algorithms:

1. Add multi-object and off-centre phantoms so localization and support metrics cannot pass merely because the target is centered.

2. Add controlled complex Gaussian noise and geometry/calibration perturbations, then run repeated seeds and report mean plus standard deviation.

3. Add a YAML scene/data schema and a Pipeline concrete so the same benchmark can be configured without editing Python.

4. Add CI and a compact example notebook for the JOSS-facing Phase-1 release.

5. Enter Phase 2 with dispersive breast/bone tissue, calibration and artifact preprocessing, and one public measured-data benchmark such as UM-BMID.

Only after those baselines are stable should TV, sparsity, plug-and-play, or AI priors be judged. A new method is valuable only when it beats the existing Born/DBIM/CSI benchmark under the same data and metrics.

## 13. Final mental model

P1-A answers **how good is the result, in several non-equivalent senses?**

P1-B answers **can every method be run fairly and reproduced from one command?**

P1-C answers **where is the scattering energy before we attempt quantitative material recovery?**

Together they change Phoenix from a collection of successful algorithm demos into the first complete platform loop:

$$
\boxed{
\text{known scene}
\rightarrow
\text{full-wave data}
\rightarrow
\{\text{DAS, Born, DBIM, CSI}\}
\rightarrow
\text{common metrics}
\rightarrow
\text{reproducible report}
}
$$
