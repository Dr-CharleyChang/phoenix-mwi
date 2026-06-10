# phoenix-mwi

二维/三维微波成像（MWI）正演与反演仿真，面向乳腺与骨密度检测研究。
配套理论笔记见 `docs/`。**当前阶段：F1（2D MoM 正演 + Mie 解析验证）。**

> Built in public · 物理优先 · 每一步都用解析解/公开基准验证。

## 仓库结构

```
phoenix-mwi/
├── mwisim/              # 核心库
│   ├── grid.py         # 网格 + 对比度 χ          (F1 §1,§3.1)
│   ├── green.py        # 2D Green 函数 G          (F1 §2)
│   ├── mom.py          # MoM 正演:D 矩阵/入射场/解/散射场 (F1 §3–§6)
│   ├── operators.py    # matrix-free A_op / AH_op  (F2，先留 stub)
│   ├── mie.py          # Mie 解析解(真值)          (F1 §7)
│   ├── metrics.py      # 误差度量 + 收敛研究        (F1 §8)
│   └── inverse/        # I 阶段:Born/BIM/DBIM/PnP (占位)
├── scripts/
│   └── run_f1.py       # F1 驱动:出两张图 + 收敛曲线
├── tests/
│   └── test_f1.py      # 自测 T1–T8(实现后转绿)
├── phantoms/           # UWCEM 体模等(占位)
├── hls/                # Zynq HLS 阶段(占位，对接 Zenith-Radar FFT 核)
└── docs/               # 教程 + 几何图
```

## 安装（Windows PowerShell，离线友好）

只需要 `numpy scipy matplotlib pytest` 四个包。**不需要 `pip install -e .`**——`scripts/run_f1.py` 已自带路径处理，测试用 `python -m pytest` 即可。

先检查是否已有这些包：

```powershell
python -c "import numpy, scipy, matplotlib, pytest; print('ok')"
```

- 打印 `ok` → 直接用，无需安装。
- 报 `ModuleNotFoundError` → 装包：
  - 有 Anaconda/Miniconda（推荐）：`conda install numpy scipy matplotlib pytest`
  - 用 pip 且联网正常：`pip install numpy scipy matplotlib pytest`
  - pip 走代理被挡（`ProxyError`/连接被拒）：加代理参数 `pip install --proxy http://用户:密码@代理地址:端口 numpy scipy matplotlib pytest`，或改用 conda，或离线 wheel。

> 不要在 WSL 里装——Windows 的 Python 和 WSL 的 Python 是两套，包不互通。统一用 Windows PowerShell。

## F1 怎么做

1. 读 `docs/F1_Tutorial_2D-MoM正演与Mie验证.md`（配合 `docs/F1_geometry_2D-TM-cylinder.svg`）。
2. 实现 `mwisim/` 里所有标了 `NotImplementedError` 的函数（每个 docstring 指向教程对应小节）。
3. 边写边跑测试，逐个转绿：
   ```bash
   pytest -q            # T1–T8 自测，全绿即 F1 通过
   ```
4. 出成果图：
   ```bash
   python scripts/run_f1.py
   ```
   生成 `docs/fig_pointwise.png`（逐点 MoM vs Mie）与 `docs/fig_convergence.png`（误差 vs 网格密度）。

> **关键纪律**：先用弱散射 `eps_r=2` 跑通、对上 Mie，再加难到 `eps_r=8`。约定统一 $e^{j\omega t}$ / $H^{(2)}$；对不上先怀疑约定（见教程 §3.4 警告）。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| **F1** | 2D MoM 正演 + Mie 验证 | ✅ 完成（pytest 7/7，逐点 3.15%，收敛单调） |
| F2 | CG-FFT 加速(matrix-free + Toeplitz) | ⏳ |
| F3 | UWCEM 体模 + Cole-Cole 多频 | ⏳ |
| I1–I4 | Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM | ⏳ |
| HLS | Zynq-7020 FFT 核加速(复用 Zenith-Radar) | ⏳ |

## 验证哲学

不和"真实测量"较劲（无体模/无临床数据），而是和**解析解**(Mie)与**公开基准**(UWCEM 体模、Institut Fresnel 实测集)对齐——纯仿真就能给出可信的"已验证"证据。
