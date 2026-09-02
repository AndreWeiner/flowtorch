<p align="center">
  <img src="https://raw.githubusercontent.com/AndreWeiner/flowtorch/main/media/flowtorch_logo.svg" alt="flowTorch logo" height="240">
</p>

# flowTorch

[![status](https://joss.theoj.org/papers/57b32d31997c90a40b3f4bdc20782e55/status.svg)](https://joss.theoj.org/papers/57b32d31997c90a40b3f4bdc20782e55)

**flowTorch** - a Python library for analysis and reduced order modeling of fluid flows

The development of flowTorch is primarily financed by the German Research Foundation (DFG) within the research program [FOR 2895](https://www.for2895.uni-stuttgart.de/) *unsteady flow and interaction phenomena at high speed stall conditions* with the primary goal to investigate flow conditions that lead to [buffeting](https://en.wikipedia.org/wiki/Aeroelasticity#Buffeting) at airfoils in the transonic flow regime.

https://user-images.githubusercontent.com/8482575/120886182-f2b78800-c5ec-11eb-9b93-efb9a139c431.mp4

The animation shows the shock buffet on a NACA-0012 airfoil at $Re=10^7$, $Ma=0.75$, and $\alpha=4^\circ$ angle of attack. The simulation was conducted with OpenFOAM; follow [this link](https://github.com/AndreWeiner/naca0012_shock_buffet) for more information about the setup.

## Why *flowTorch*?

The *flowTorch* project was started to make the analysis and modeling of fluid data **easy** and **accessible** to everyone. The library design intends to strike a balance between **usability** and **flexibility**. Instead of a monolithic, black-box analysis tool, the library offers modular components that allow assembling custom analysis and modeling workflows with ease. *flowTorch* helps to fuse data from a wide range of file formats typical for fluid flow data, for example, to compare experimental and simulation data. The available analysis and modeling tools are rigorously tested and demonstrated on a variety of different fluid flow datasets. Moreover, one can significantly accelerate the entire process of accessing, cleaning, analyzing, and modeling fluid flow data by starting with one of the pipelines available in the *flowTorch* [documentation](https://flowtorch.readthedocs.io/en/latest/).

To get a first impression of what working with *flowTorch* looks like, the code snippet below shows part of a pipeline for performing a dynamic mode decomposition (DMD) of a transient *OpenFOAM* simulation.

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

Currently, the following sub-packages are under active development. Note that some components are not yet available in the public release because further developments and testing are required:

| package | content |
| :------ | :-------|
|flowtorch.data | data loading, domain reduction (masked selection), outlier removal/masking |
| flowtorch.analysis | algorithms for dimensionality reduction and modal analysis (e.g., SVD, DMD, MSSA) |
| flowtorch.rom | reduced-order modeling |
| flowtorch.visualization | convenience functions for comparative plots and animations |

*flowTorch* uses the [PyTorch](https://github.com/pytorch/pytorch) library as a backend for data structures, data types, and linear algebra operations on CPU and GPU. Some cool features of *flowTorch* include:

- data accessors return PyTorch tensors, which can be used directly within your favorite machine learning library, e.g., *PyTorch*, *scikit-learn*, or *TensorFlow*
- most algorithms run on CPU as well as on GPU
- mixed-precision operations (single/double); switching to single precision makes your life significantly easier when dealing with large datasets
- user-friendly Python library that integrates easily with popular tools and libraries like *Jupyterlab*, *Matplotlib*, *Pandas*, or *Numpy*
- a rich tutorial collection to help you get started
- interfaces to common data formats like [OpenFOAM](https://www.openfoam.com/), batched NumPy output from the [`foamToNumpy`](https://github.com/tanujravi/numpyToFoam) function object, [VTK](https://vtk.org/) (for Flexi and SU2), [TAU](https://www.dlr.de/en/as/research-and-transfer/software-solutions/aerodynamics/software-tau), [iPSP](https://www.dlr.de/en/as/about-us/departments/experimental-methods/pressure-sensitive-paint-psp), and CSV (for DaVis PIV data and raw OpenFOAM output)

*flowTorch* can also be used easily in combination with existing Python packages for analysis and reduced-order modeling thanks to the interoperability between PyTorch and NumPy. Great examples are (by no means a comprehensive list):

- [PyDMD](https://github.com/mathLab/PyDMD) - Python dynamic mode decomposition
- [PySINDy](https://github.com/dynamicslab/pysindy) - sparse identification of nonlinear dynamical systems from data
- [PyKoopman](https://github.com/dynamicslab/pykoopman) - data-driven approximations of the Koopman operator

## Getting started

Install the latest stable release from PyPI:
```
pip install flowtorch-fluid

# to uninstall flowTorch, run
pip uninstall flowtorch-fluid
```
The default installation includes all dependencies required by
`flowtorch.analysis` except for the optional iPSP explorer. Install additional
functionality with an optional extra:
```
# data loaders, reduced-order models, or the iPSP explorer
pip install "flowtorch-fluid[data]"
pip install "flowtorch-fluid[rom]"
pip install "flowtorch-fluid[psp]"

# all optional functionality
pip install "flowtorch-fluid[all]"
```
Extras can be combined, for example
`pip install "flowtorch-fluid[data,rom]"`.

The PyPI distribution is named `flowtorch-fluid`, while the Python package and
imports remain `flowtorch`.

To install the latest development version directly from GitHub, run:
```
pip install "flowtorch-fluid[all] @ git+https://github.com/AndreWeiner/flowtorch.git"
```
Alternatively, clone the repository:
```
git clone git@github.com:AndreWeiner/flowtorch.git
cd flowtorch
```
and install it in editable mode with the desired optional dependencies:
```
pip install -e ".[all]"
```
Installing all flowTorch dependencies requires significant disk space. Replace
`all` with `data`, `rom`, or `psp` when only part of the optional functionality
is needed.

To get an overview of what *flowTorch* can do for you, have a look at the [online documentation](https://flowtorch.readthedocs.io/en/latest/). The examples presented in the online documentation are also contained in this repository. In fact, the documentation is a static version of several [Jupyter notebooks](https://jupyter.org/) with end-to-end analyses. If you are interested in an interactive version of one particular example, navigate to `./docs/source/notebooks` and run `jupyter lab`. Note that to execute some of the notebooks, the **corresponding datasets are required**. The datasets can be downloaded [here](https://datashare.tu-dresden.de/s/rekLnoqzRCp9zk9) (~2.6GB). If the data are only required for unit testing, a reduced dataset may be downloaded [here](https://datashare.tu-dresden.de/s/dr7gBPSdeyXQrgd) (~411MB). Download the data into a directory of your choice and navigate into that directory. To extract the archive, run:
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

`TecplotDataloader` is supported on Linux and macOS only. ParaView is required
only by this loader, which uses ParaView's
`VisItTecplotBinaryReader` to access binary
[Tecplot](https://www.tecplot.com/) files. Download and install a ParaView build
that includes the VisItBridge readers, then locate the `pvpython` executable
included in that installation.

ParaView runs in an isolated subprocess. Its bundled Python packages are not
added to the flowTorch interpreter, so ParaView does not need to match the
Python version of the flowTorch environment. Do not add ParaView's
`site-packages` directory to the global `PYTHONPATH`, and do not remove or
replace packages inside the ParaView installation.

Pass the executable path when constructing the loader:

```python
from flowtorch.data import TecplotDataloader

loader = TecplotDataloader.from_tau(
    path,
    "alfa16.surface.pval.unsteady_",
    pvpython="/opt/ParaView/bin/pvpython",
)
```

Alternatively, configure the executable once:

```bash
export FLOWTORCH_PVPYTHON=/opt/ParaView/bin/pvpython
```

When `pvpython` is already on `PATH`, no explicit configuration is required.
The selected ParaView installation can be checked with:

```bash
"${FLOWTORCH_PVPYTHON:-pvpython}" -c \
  "from paraview.simple import VisItTecplotBinaryReader; print('Tecplot reader available')"
```

The loader keeps one `pvpython` worker running and transfers arrays directly
over a binary inter-process channel; it does not create temporary data files.
Close the worker explicitly when finished:

```python
loader.close()
```

The loader can also be used as a context manager:

```python
with TecplotDataloader.from_tau(
    path,
    "alfa16.surface.pval.unsteady_",
    pvpython="/opt/ParaView/bin/pvpython",
) as loader:
    density = loader.load_snapshot("density", loader.write_times)
```

## Development
### Documentation

Build the flowTorch documentation in an isolated tox environment:
```
tox -e docs
```
Tox installs Sphinx and the documentation dependencies automatically. The
generated HTML documentation is written to `docs/build/html`; open
`docs/build/html/index.html` in a browser to view it. The build treats Sphinx
warnings as errors so that documentation problems are caught locally.

### Packaging

Build the source distribution and wheel and validate their PyPI metadata with:
```
tox -e package
```
The generated artifacts are written to `dist/`. The PyPI distribution is named
`flowtorch-fluid`, while the Python import package remains `flowtorch`.

### Unit testing
The test suite is located in the top-level `tests` directory. To install the development testing tools, run:
```
pip install -r requirements-dev.txt
```
To run the default test suite with the active Python interpreter, execute:
```
pytest
```
Tests that require the flowTorch datasets are marked as integration tests and are skipped unless the datasets are downloaded and referenced as described in the previous section. To run only tests that do not require external datasets, execute:
```
pytest -m "not integration"
```
You can also execute all tests for one test group, e.g., data:
```
pytest tests/data
```
or run individual test modules, e.g.,
```
pytest tests/data/test_foam_dataloader.py
```

To run the dataset-free tests with multiple Python versions, use tox:
```
tox
```
The default tox configuration runs `py310`, `py312`, and `py314` environments and skips Python versions that are not installed locally. You can run a single environment with:
```
tox -e py312
```
Additional pytest arguments can be passed after `--`, for example:
```
tox -e py312 -- tests/data/test_utils.py
```
To run dataset-dependent integration tests through tox, make sure `FLOWTORCH_DATASETS` is set and pass the marker explicitly:
```
tox -e py312 -- -m integration
```

### Code formatting

Python code is formatted with [Black](https://black.readthedocs.io/) using its
default line length of 88 characters. Tox installs the pinned Black version in
an isolated environment, so no separate Black installation is required. Format
the package and tests with:
```
tox -e format
```
To format only specific files or directories, pass them after `--`, for example:
```
tox -e format -- flowtorch/analysis tests/analysis
```
Check formatting without changing any files with:
```
tox -e format-check
```

### Code linting

Python code is linted with [Ruff](https://docs.astral.sh/ruff/). Tox installs
the pinned Ruff version in an isolated environment. Lint the package and tests
with:
```
tox -e lint
```
To lint only specific files or directories, pass them after `--`, for example:
```
tox -e lint -- flowtorch/analysis tests/analysis
```

### Type checking

Python type annotations are checked with [mypy](https://mypy.readthedocs.io/).
Tox installs the pinned mypy version in an isolated environment, so no separate
installation is required. Check the `flowtorch` package with:
```
tox -e type-check
```
To check only specific files or directories, pass them after `--`, for example:
```
tox -e type-check -- flowtorch/analysis/svd.py
```
The shared type-checking options are defined in `mypy.ini`.

## Getting help

If you encounter any issues using *flowTorch* or if you have any questions regarding current and future development plans, please use the repository's [issue tracker](https://github.com/AndreWeiner/flowtorch/issues). Consider the following steps before and when opening a new issue:

0. Have you searched for similar issues that may have been already reported? The issue tracker has a *filter* function to search for keywords in open issues.
1. Click on the green *New issue* button in the upper right corner and describe your problem in as much detail as possible. The issue should state what **the problem** is, what the **expected behavior** should be, and, maybe, suggest a **solution**. Note that you can also attach files or images to the issue.
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

For a list of scientific works relying on flowTorch, refer to [Google Scholar](https://scholar.google.de/scholar?oi=bibs&hl=de&cites=3453204270157785607&as_sdt=5).

## License

*flowTorch* is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/AndreWeiner/flowtorch/blob/main/LICENSE) file for more information.
