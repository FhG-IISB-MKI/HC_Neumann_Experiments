# Hard-constraining Neumann boundary conditions in PINNs

This repository provides the source code used for the numerical experiments in the workshop paper [Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings](https://openreview.net/forum?id=jKdZsWdRLZ) (including the environment used to run it).

## Citation
```
@inproceedings{straub2025hard,
    title = {Hard-constraining Neumann boundary conditions in physics-informed neural networks via Fourier feature embeddings},
    author = {Christopher Straub and Philipp Brendel and Vlad Medvedev and Andreas Rosskopf},
    booktitle = {ICLR 2025 Workshop on Machine Learning Multiscale Processes},
    year = {2025},
    url = {https://openreview.net/forum?id=jKdZsWdRLZ}
}
```

## Notes
The code is based on the popular PINN framework [DeepXDE](https://github.com/lululxvi/deepxde). The purpose of the code is only to demonstrate the efficacy for a rather simple forward problem. Consequently, the code has been written in a fairly straightforward way. We encourage readers to test the versatile method described in the paper for problems that extend beyond the diffusion problem considered in the code.

## Contact
Please contact [Christopher Straub](mailto:christopher.straub@iisb.fraunhofer.de) in case of any question, comments, etc. regarding the code or the paper.
