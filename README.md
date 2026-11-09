# D-BRANN

[![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/task-sparse--IMU%20pose%20estimation-blue)](https://wearnet.github.io/D-BRANN/)
[![Docs](https://img.shields.io/badge/docs-wearnet.github.io%2FD--BRANN-informational)](https://wearnet.github.io/D-BRANN/)
[![Paper](https://img.shields.io/badge/paper-Electronics%202026-success)](https://doi.org/10.3390/electronics15215112)
[![Dataset](https://img.shields.io/badge/dataset-Zenodo-1682d4)](https://doi.org/10.5281/zenodo.15778950)

**D-BRANN** is a distributed bidirectional recurrent architecture for full-body human pose estimation from six sparse inertial measurement units (IMUs).

D-BRANN builds on the excellent work of [TransPose](https://github.com/Xinyu-Yi/TransPose) (Yi et al.), whose centralized architecture and translation modules (Trans-B1, Trans-B2) form the foundation this project extends into a distributed, branch-based design.

📖 **Full documentation:** [wearnet.github.io/D-BRANN](https://wearnet.github.io/D-BRANN/) — architecture, environment setup, checkpoints, data preparation, training, evaluation, and Xsens/Unity integration.

📄 **Paper:** K. Niño-Tejada, J. L. Rivas-Caicedo, L. Saldaña-Aristizábal, and J. F. Patarroyo-Montenegro, "[D-BRANN: A Distributed Bidirectional Recurrent Neural Network for Full-Body Joint Angle Estimation from Sparse Inertial Measurement Units](https://doi.org/10.3390/electronics15215112)," *Electronics*, vol. 15, no. 21, p. 5112, 2026.

🗂️ **Dataset:** [MoCap and IMU Dataset for Upper Limb Joint Angle Estimation](https://doi.org/10.5281/zenodo.15778950), Zenodo (2025).

---

## Acknowledgments

This research received financial support from the NSF CAREER Intelligent Biomarker Analysis based on Wearable Distributed Computing under Grant No. OAC-2439345.

---

## License

The source code is released under the [GNU General Public License v3.0](LICENSE), as it builds on [TransPose](https://github.com/Xinyu-Yi/TransPose), which is distributed under the same license.

Third-party assets are **not** covered by this license and are subject to their own terms: the SMPL model, the AMASS, DIP-IMU and TotalCapture datasets, and the original TransPose checkpoint. They are not redistributed here and must be obtained from their official sources. The MoCap and IMU dataset is released on Zenodo under CC BY 4.0.

---

## Citation

If you use D-BRANN in your research, please cite the paper ([DOI: 10.3390/electronics15215112](https://doi.org/10.3390/electronics15215112)):

```bibtex
@article{ninotejada2026dbrann,
  author  = {Ni{\~n}o-Tejada, Kevin and Rivas-Caicedo, Jhonathan L. and Salda{\~n}a-Aristiz{\'a}bal, Laura and Patarroyo-Montenegro, Juan F.},
  title   = {{D-BRANN}: A Distributed Bidirectional Recurrent Neural Network for Full-Body Joint Angle Estimation from Sparse Inertial Measurement Units},
  journal = {Electronics},
  year    = {2026},
  volume  = {15},
  number  = {21},
  pages   = {5112},
  doi     = {10.3390/electronics15215112}
}
```

If you use the dataset, please also cite it ([DOI: 10.5281/zenodo.15778950](https://doi.org/10.5281/zenodo.15778950)):

```bibtex
@misc{ninotejada2025mocap,
  author    = {Ni{\~n}o-Tejada, Kevin and Salda{\~n}a-Aristiz{\'a}bal, Laura and Rivas-Caicedo, Jhonathan L. and Patarroyo-Montenegro, Juan F.},
  title     = {{MoCap and IMU Dataset for Upper Limb Joint Angle Estimation}},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15778950}
}
```
