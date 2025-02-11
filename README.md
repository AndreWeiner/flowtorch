![FOR2895Logo](media/flowTorch_Logo_Wide.png)

# flowTorch

[![status](https://joss.theoj.org/papers/57b32d31997c90a40b3f4bdc20782e55/status.svg)](https://joss.theoj.org/papers/57b32d31997c90a40b3f4bdc20782e55)

**flowTorch** - a Python library for analysis and reduced order modeling of fluid flows

The development of flowTorch is primarily financed by the German Research Foundation (DFG) within the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/) *unsteady flow and interaction phenomena at high speed stall conditions* with the primary goal to investigate flow conditions that lead to [buffeting](https://en.wikipedia.org/wiki/Aeroelasticity#Buffeting) at airfoils in the transonic flow regime.

https://user-images.githubusercontent.com/8482575/120886182-f2b78800-c5ec-11eb-9b93-efb9a139c431.mp4

The animation shows the shock buffet on a NACA-0012 airfoil at $Re=10^7$, $Ma=0.75$, and $\alpha=4^\circ$ angle of attack. The simulation was conducted with OpenFOAM; follow [this link](https://github.com/AndreWeiner/naca0012_shock_buffet) for more information about the setup.

## Why *flowTorch*?

The *flowTorch* project was started to make the analysis and modeling of fluid data **easy** and **accessible** to everyone. The library design intends to strike a balance between **usability** and **flexibility**. Instead of a monolithic, black-box analysis tool, the library offers modular components that allow assembling custom analysis and modeling workflows with ease. *flowTorch* helps to fuse data from a wide range of file formats typical for fluid flow data, for example, to compare experiments simulations. The available analysis and modeling tools are rigorously tested and demonstrated on a variety of different fluid flow datasets. Moreover, one can significantly accelerate the entire process of accessing, cleaning, analyzing, and modeling fluid flow data by starting with one of the pipelines available in the *flowTorch* [documentation](https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/index.html).

To get a first impression of how working with *flowTorch* looks like, the code snippet below shows part of a pipeline for performing a dynamic mode decomposition (DMD) of a transient *OpenFOAM* simulation.

```
import torch as pt
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis.dmd import DMD

path = DATASETS["of_cylinder2D_binary"]
loader = FOAMDataloader(path)

# select a subset of the available snapshots
times = loader.write_times
window_times = [time for time in times if float(time) >= 4.0]

# load vertices, discard z-coordinate, and create a mask
vertices = loader.vertices[:, :2]
mask = mask_box(vertices, lower=[0.1, -1], upper=[0.75, 1])

# assemble the data matrix
data_matrix = pt.zeros((mask.sum().item(), len(window_times)), dtype=pt.float32)
for i, time in enumerate(window_times):
    # load the vorticity vector field, take the z-component [:, 2], and apply the mask
    data_matrix[:, i] = pt.masked_select(loader.load_snapshot("vorticity", time)[:, 2], mask)

# perform DMD
dmd = DMD(data_matrix, rank=19)
# analyze dmd.modes or dmd.eigvals
# ...
```

Currently, the following sub-packages are under active development. Note that some of the components are not yet available in the public release because further developments and testing are required:

| package | content |
| :------ | :-------|
|flowtorch.data | data loading, domain reduction (masked selection), outlier removal |
| flowtorch.analysis | algorithms for dimensionality reduction and modal analysis (e.g., SVD, DMD, MSSA) |
| flowtorch.rom | reduced-order modeling (CNM) |

*flowTorch* uses the [PyTorch](https://github.com/pytorch/pytorch) library as a backend for data structures, data types, and linear algebra operations on CPU and GPU. Some cool features of *flowTorch* include:

- data accessors return PyTorch tensors, which can be used directly within your favorite machine learning library, e.g., *PyTorch*, *SkLearn* or *Tensorflow*
- most algorithms run on CPU as well as on GPU
- mixed-precision operations (single/double); switching to single precision makes your life significantly easier when dealing with large datasets
- user-friendly Python library that integrates easily with popular tools and libraries like *Jupyterlab*, *Matplotlib*, *Pandas*, or *Numpy*
- a rich tutorial collection to help you getting started
- interfaces to common data formats like [OpenFOAM](https://www.openfoam.com/), [VTK](https://vtk.org/) (for Flexi and SU2), [TAU](https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/), [iPSP](https://www.dlr.de/as/en/desktopdefault.aspx/tabid-183/251_read-13334/), CSV (for DaVis PIV data and raw OpenFOAM output)

*flowTorch* can be also used easily in combination with existing Python packages for analysis and reduced-order modeling thanks to the interoperability between PyTorch and NumPy. Great examples are (by no means a comprehensive list):

- [PyDMD](https://github.com/mathLab/PyDMD) - Python dynamic mode decomposition
- [PySINDy](https://github.com/dynamicslab/pysindy) - sparse identification of nonlinear dynamical systems from data
- [PyKoopman](https://github.com/dynamicslab/pykoopman) - data-driven approximations of the Koopman operator

## Getting started

The easiest way to install *flowTorch* is as follows (use the development branch *aweiner* for access to the latest developments):
```
# install via pip
pip3 install git+https://github.com/FlowModelingControl/flowtorch
# or install a specific branch, e.g., aweiner
pip3 install git+https://github.com/FlowModelingControl/flowtorch.git@aweiner

# to uninstall flowTorch, run
pip3 uninstall flowtorch
```
Alternatively, you can also clone the repository manually by running
```
git clone git@github.com:FlowModelingControl/flowtorch.git
```
and install the dependencies listed in *requirements.txt*:
```
pip3 install -r requirements.txt
```
Installing all flowTorch dependencies requires a significant amount of disk space. When using isolated subpackages, one can also install the dependencies manually (by trial-and-error). To load the library package from within a Python script file or a Jupyter notebook, add the path to the cloned repository as follows:
```
import sys
sys.path.insert(0, "/path/to/repository")
```

To get an overview of what *flowTorch* can do for you, have a look at the [online documentation](https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/index.html). The examples presented in the online documentation are also contained in this repository. In fact, the documentation is a static version of several [Jupyter labs](https://jupyter.org/) with start-to-end analyses. If you are interested in an interactive version of one particular example, navigate to `./docs/source/notebooks` and run `jupyter lab`. Note that to execute some of the notebooks, the **corresponding datasets are required**. The datasets can be downloaded [here](https://cloud.tu-braunschweig.de/s/sJYEfzFG7yDg3QT) (~2.6GB). If the data are only required for unit testing, a reduced dataset may be downloaded [here](https://cloud.tu-braunschweig.de/s/b9xJ7XSHMbdKwxH) (~411MB). Download the data into a directory of your choice and navigate into that directory. To extract the archive, run:
```
# full dataset
tar xzf datasets_29_10_2021.tar.gz
# reduced dataset
tar xzf datasets_minimal_29_10_2021.tar.gz
```
To tell *flowTorch* where the datasets are located, define the `FLOWTORCH_DATASETS` environment variable:
```
# add export statement to bashrc; assumes that the extracted 'datasets' or 'datasets_minimal'
# folder is located in the current directory
# full dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets/\"" >> ~/.bashrc
# reduced dataset
echo "export FLOWTORCH_DATASETS=\"$(pwd)/datasets_minimal/\"" >> ~/.bashrc
# reload bashrc
. ~/.bashrc
```

## Installing ParaView

**Note:** the following installation of ParaView is only necessary if the *TecplotDataloader* is needed.

*flowTorch* uses the ParaView Python module for accessing [Tecplot](https://www.tecplot.com/) data. When installing ParaView, special attention must be paid to the installed Python and VTK versions. Therefore, the following manual installation is recommend instead of using a standard package installation of ParaView.

1. Determine the version of Python:
```
python3 --version
# example output
Python 3.8.10
```
2. Download the ParaView binaries according to your Python version from [here](https://www.paraview.org/download/). Note that you may have to use an older version ParaView to match your Python version.
3. Install the ParaView binaries, e.g., as follows:
```
# optional: remove old package installation if available
sudo apt remove paraview
# replace the archive's name if needed in the commands below
sudo mv ParaView-5.9.1-MPI-Linux-Python3.8-64bit.tar.gz /opt/
cd /opt
sudo tar xf ParaView-5.9.1-MPI-Linux-Python3.8-64bit.tar.gz
sudo rm ParaView-5.9.1-MPI-Linux-Python3.8-64bit.tar.gz
cd ParaView-5.9.1-MPI-Linux-Python3.8-64bit/
# add path to ParaView binary and Python modules
echo export PATH="\$PATH:$(pwd)/bin" >> ~/.bashrc
echo export PYTHONPATH="\$PYTHONPATH:$(pwd)/lib/python3.8/site-packages" >> ~/.bashrc
```
In case of version conflicts between Python packages coming with ParaView and local versions of these packages, the following options exist:
1. go to your ParaView installation and manually delete or rename the affected packages; the packages are located at */path/to/ParaView/lib/python3.8/site-packages*
2. use *pvpython*, a modified Python interpreter shipped with ParaView and add a virtual environment containing flowTorch but not the conflicting packages (see [Using pvpython and virtualenv](https://www.kitware.com/using-pvpython-and-virtualenv/))

## Development
### Documentation

To build the flowTorch documentation, the following additional packages are required:
```
pip3 install sphinx sphinx_rtd_theme nbsphinx recommonmark
```
To build the HTML version of the API documentation, navigate to `./docs` and run:
```
make html
```

### Unit testing
All sub-packages contain unit tests, which require the installation of PyTest:
```
pip3 install pytest
```
Moreover, the flowTorch datasets must be downloaded and referenced as described in the previous section.
To run all unit tests of all sub-packages, execute:
```
pytest flowtorch
```
You can also execute all tests in a sub-package, e.g., data
```
pytest flowtorch/data
```
or run individual test modules, e.g.,
```
pytest flowtorch/data/test_FOAMDataloader.py
```

## Getting help

If you encounter any issues using *flowTorch* or if you have any questions regarding current and future development plans, please use the repository's [issue tracker](https://github.com/FlowModelingControl/flowtorch/issues). Consider the following steps before and when opening a new issue:

0. Have you searched for similar issues that may have been already reported? The issue tracker has a *filter* function to search for keywords in open issues.
1. Click on the green *New issue* button in the upper right corner and describe your problem as detailed as possible. The issue should state what **the problem** is, what the **expected behavior** should be, and, maybe, suggest a **solution**. Note that you can also attach files or images to the issue.
2. Select a suitable label from the drop-down menu called *Labels*.
3. Click on the green *Submit new issue* button and wait for a reply.

## Reference

If *flowTorch* aids your work, you may support the project by referencing the following article:

```
@article{Weiner2021,
doi = {10.21105/joss.03860},
url = {https://doi.org/10.21105/joss.03860},
year = {2021},
publisher = {The Open Journal},
volume = {6},
number = {68},
pages = {3860},
author = {Andre Weiner and Richard Semaan},
title = {flowTorch - a Python library for analysis and reduced-order modeling of fluid flows},
journal = {Journal of Open Source Software}
} 
```

For a list of scientific works relying on flowTorch, refer to [this list](references.md).

## License

*flowTorch* is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/FlowModelingControl/flowtorch/blob/main/LICENSE) file for more information.

