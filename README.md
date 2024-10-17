# cpDistiller

## Overview

cpDistiller (Triple-effect correction for Cell Painting data with contrastive and domain-adversarial learning) is specifically designed for Cell Painting data to map inputs into the low-dimensional embedding spaces that correct multiple technical effects, such as batch, row, and column effects, while capturing true biological signals.


![](./images/cpDistiller.jpg)

## Install

### Step1

Download the package from Github and install it locally.

```bash
git clone https://github.com/Cell-Painting/cpDistiller
cd cpDistiller
```

### Step2

Create a conda environment, and then activate it as follows in terminal.   

```bash
conda env create -f environment.yml
conda activate cpDistiller
```

### Step3

Download the environment packages required for the pre-trained model.

```bash
cd deepcell_release
python setup.py install
pip install deepcell_toolbox==0.12.1
```
### Step4

You can install cpDistiller by the following command:

```bash
cd ..
pip install -e .
```
## Quick Start
More details could be found in [cpDistiller documents](https://cpdistiller.readthedocs.io/).

## License
We used the pre-trained Mesmer for transfer learning. Mesmer is licensed under a Modified Apache License 2.0 for non-commercial, academic use only. See [LICENSE](https://github.com/Cell-Painting/cpDistiller/blob/main/LICENSE-MODIFIED-APACHE-2.0) for full details.

Other parts is licensed under MIT License.

## Acknowledgement

We sincerely thank the authors of the following open-source projects:

- [Mesmer](https://github.com/vanvalenlab/intro-to-deepcell, https://doi.org/10.1038/s41587-021-01094-0)