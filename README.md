# `bahamas`-galactic-foreground-data-release

Data release supporting:
_The first year of LISA Galactic foreground_

R.Buscicchio, F.Pozzoli, D.Chirico, A.Sesana. 
[arXiv: XXX.YYY](https://arxiv.org/abs/XXX.YYY).

## Credits

You are welcome to use this dataset in your research. We kindly ask you to cite the paper above.

If you want to cite specifically the data release, its DOI is: 

[![DOI](https://zenodo.org/badge/901868271.svg)](https://doi.org/XXYYZZ/zenodo.XXYYZZ)

And the content is mapped in [github release page](https://github.com/RiccardoBuscicchio/bahamas-galactic-foreground-release/releases). 


## Data

Minimal data behind figures are provided within the repository in the `./data/` subfolder.

## Content

In `./notebooks`, we provide a jupyter notebook for the following figures:

- Figure X: `./notebooks/FigureX.ipynb`
- Figure Y: `./notebooks/FigureY.ipynb`
- Figure Z: `./notebooks/FigureZ.ipynb`

Figure A, Figure B, Figure C and Figure D require large datasets to be produced.
Feel free to get in touch with the authors, should you need access to the data.
 
## Requirements

Feel free to use `./bahamasgf.yaml` to create a conda environment to reproduce our figures, with 
```bash
conda env create -f bahamasgf.yaml
```
Then, if working with jupyter notebooks, install the kernel with:
```bash
python -m ipykernel install --user --name bahamasgf --display-name "bahamasgf"
```
