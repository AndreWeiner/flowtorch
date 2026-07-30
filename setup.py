#!/bin/python3

from setuptools import setup

exec(open("flowtorch/version.py").read())

setup(
    name="flowTorch",
    version=__version__,
    description="Analysis and reduced-order modeling of fluid flows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="flowTorch contributors",
    author_email="a.weiner@tu-braunschweig.de",
    url="https://github.com/FlowModelingControl/flowtorch",
    packages=["flowtorch", "flowtorch.data", "flowtorch.analysis", "flowtorch.rom"],
    license="GPL-v3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Unix Shell",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    install_requires=[
        "h5py",
        "netCDF4",
        "matplotlib",
        "numpy",
        "numpy-stl",
        "pandas",
        "plotly",
        "torch >= 1.9",
        "jupyterlab",
        "scikit-learn",
        "vtk",
    ],
)
