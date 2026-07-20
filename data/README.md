# External measured data

Large public datasets are not committed to this repository. The Phase-2A driver stores the pinned UM-BMID Gen-One archive and extracted files under `data/external/um_bmid/`, which is ignored by Git.

```bash
python scripts/run_p2_um_bmid.py --download --sample-id 1
```

The driver verifies the official Zenodo archive size and MD5, safely extracts it, checks the extracted members against the verified ZIP, and only then opts into loading the official pickle files. See [the Phase-2 tutorial](../docs/P2_Tutorial_Measured-Data-from-zero-to-100.md) for the trust model, schema, attribution, and benchmark scope.
