# hls/

Zynq-7020 HLS 加速阶段（最后做）。

策略：PS(ARM) 跑 DBIM 外层 + 控制；PL(FPGA) 用 HLS 加速一个热点核——最现实的是
**CG-FFT 里的 FFT + 复数逐元素乘**（$\mathbf G\mathbf v$）。这个 FFT 核与 **Zenith-Radar OS** 的
1D/2D-FFT 核同源，可直接复用 → 两个项目在此合流。

前置：先把 Python 端 F1/F2 跑通并验证，作为 HLS 的 golden reference（位真值对拍）。
7020 资源有限（220 DSP48 / 4.9 Mb BRAM）：完整 3D 不可能，定位为 2D 概念验证 / 边缘加速 demo。
