# phantoms/

Digital phantoms and public datasets (not tracked in git; download separately).

- **UWCEM numerical breast phantoms** (University of Wisconsin, Zastrow/Hagness):
  voxel-wise tissue labels paired with Lazebnik Cole-Cole parameters. Used in F3
  and for inversion validation.
- **Institut Fresnel measured scattering datasets**: 2D/3D experimental scattered
  fields, for validation against *real* measurements (optional, advanced).

Place downloaded data here (`*.h5` / `*.mat` are gitignored). F1 does not need any
phantom — its ground truth is the analytic Mie series.
