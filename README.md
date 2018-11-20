# Light curve classifier (LCC)

Light curve classifier (LCC) is an approach to classify lightcurves using convolutional neural networks.


## Install LCC

### Clone Repository with Git

Clone the CTLearn repository:

```bash
cd </installation/path>
git clone https://github.com/TjarkMiener/LightCurveClassifier.git
```

### Install Package with Anaconda

Next, download and install [Anaconda](https://www.anaconda.com/download/), or, for a minimal installation, [Miniconda](https://conda.io/miniconda.html). Create a new conda environment that includes all the dependencies for LCC:

```bash
conda env create -f </installation/path>/LightCurveClassifier/environment-<MODE>.yml
```

where `<MODE>` is either 'cpu' or 'gpu', denoting the TensorFlow version to be installed. If installing the GPU version of TensorFlow, verify that your system fulfills all the requirements [here](https://www.tensorflow.org/install/install_linux#NVIDIARequirements).

Finally, install LCC into the new conda environment with pip:

```bash
source activate lcc
cd </installation/path>/LightCurveClassifier
pip install --upgrade .
```
NOTE for developers: If you wish to fork/clone the respository and make changes to any of the ctlearn modules, the package must be reinstalled for the changes to take effect.

### Dependencies

- Python 3.6.5
- TensorFlow
- Keras
- NumPy
- Astropy
- PyTables
- PyYAML
- SciPy
- Libraries used only in plotting scripts (optional)
- Matplotlib
- Pillow
- Scikit-learn


## Uninstall LCC

### Remove Anaconda Environment

First, remove the conda environment in which LCC is installed and all its dependencies:

```bash
conda remove --name lcc --all
```

### Remove LCC

Next, completely remove LCC from your system:

```bash
rm -rf </installation/path>/LightCurveClassifier
```

