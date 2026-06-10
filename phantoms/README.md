# phantoms/

数字体模与公开数据集（不纳入 git，单独下载）。

- **UWCEM 数字乳腺体模**（Wisconsin，Zastrow/Hagness）：voxel-wise 组织标签，配 Lazebnik Cole-Cole 参数。用于 F3 及反演验证。
- **Institut Fresnel 实测散射数据集**：2D/3D 实验散射场，用于对"真实测量"验证（可选，进阶）。

下载后放本目录（`*.h5` / `*.mat` 已在 .gitignore 忽略）。F1 阶段用不到——F1 的真值是 Mie 解析解，无需任何体模。
