# `bahamas` galactic-foreground-data-release

Data release supporting:
_The first year of LISA Galactic foreground_

R.Buscicchio, F.Pozzoli, D.Chirico, A.Sesana. 
[arXiv: XXX.YYY](https://arxiv.org/abs/XXX.YYY).

## Credits

You are welcome to use this dataset in your research. We kindly ask you to cite the paper above.

If you want to cite specifically the data release, its DOI is: 

[![DOI](https://zenodo.org/badge/901868271.svg)](https://doi.org/XXYYZZ/zenodo.XXYYZZ)

And the content is mapped in [github release page](https://github.com/RiccardoBuscicchio/bahamas-galactic-foreground-release/releases). 


# Git LFS 
Some of the data files are large and tracked with Git LFS.
To obtain them, please make sure you have Git LFS installed, and clone the repository with:
```bash
git lfs install
git clone <repo-url>
```

The only exception are the `yorsh_1b-1_training_sobhb.h5` input data provided by the LISA Data Challenge (LDC) team, which you can download from the [LDC website](https://lisa-ldc.in2p3.fr/).
## Data

Minimal data behind figures are provided within the repository in the `./data/` subfolder.
In addition, we release posteriors samples files and injection/inference `.yaml` config files for reproducibility. 
The `bahamas` code version associated to the publication matches the version released under the `v1.0.0-gf` tag. 
More details available here: [bahamas release page](https://github.com/FedericoPozzoli/bahamas/tree/v1.0.0-gf).

## Content

In `./notebooks`, we provide a jupyter notebook for the following figures:

- Figure 1: `./notebooks/Figure1.ipynb`
- Figure 3: `./notebooks/Figure3.ipynb`
- Figure 4: `./notebooks/Figure4.ipynb`
- Figure 5: `./notebooks/Figure5.ipynb`
- Figure 6 (left panel): `./notebooks/Figure6Left.ipynb`
- Figure 6 (right panel): `./notebooks/Figure6Right.ipynb`
- Figure 7: `./notebooks/Figure7.ipynb`
- Figure 8: `./notebooks/Figure8.ipynb`
- Figure 9: `./notebooks/Figure9.ipynb`
- Figure 10: `./notebooks/Figure10.ipynb`
- Figure A1: `./notebooks/FigureA1.ipynb`
- Figure A2: `./notebooks/FigureA2.ipynb`
- Figure A3: `./notebooks/FigureA3.ipynb`

 
## Requirements

Feel free to use `./bahamasgf.yaml` to create a conda environment to reproduce our figures, with 
```bash
conda env create -f bahamasgf.yaml
```
Then, if working with jupyter notebooks, install the kernel with:
```bash
python -m ipykernel install --user --name bahamasgf --display-name "bahamasgf"
```

## Contributions 
Feel free to open issues or pull requests if you find any problem or want access to deeper data behind figures!