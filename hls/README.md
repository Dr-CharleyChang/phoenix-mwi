# hls/

Zynq-7020 HLS acceleration stage (done last).

Strategy: the PS (ARM) runs the outer DBIM loop and control flow; the PL (FPGA)
accelerates one hot kernel via HLS — most realistically the **FFT + complex
element-wise multiply** inside CG-FFT (the $\mathbf{G}\mathbf{v}$ operator). That FFT
core is the same one used by the **Zenith-Radar OS** 1D/2D-FFT pipeline, so the two
projects converge here.

Prerequisite: get the Python F1/F2 pipeline working and validated first, to serve as
the golden reference (bit-true comparison) for the HLS implementation. The 7020 is
resource-limited (220 DSP48 / 4.9 Mb BRAM): full 3D is out of reach, so this is a
2D proof-of-concept / edge-acceleration demo.
