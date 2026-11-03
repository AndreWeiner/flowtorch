flowTorch Documentation
=======================

flowTorch provides tools for loading flow data, computing statistics and modal
analyses, and building reduced-order models. Choose a format guide for data
access, an algorithm guide to learn a method, or a worked example that applies
these tools to a dataset.

.. toctree::
   :maxdepth: 1
   :caption: Getting started and reference

   overview/glossary
   provenance

Data formats and conversion
---------------------------

Load, convert and export fields while preserving geometry and field association.

.. container:: homepage-section-links

   .. toctree::
      :maxdepth: 1
      :caption: Data formats and conversion

      ensight_io
      hdf5_workflow
      foam_hdf5

Algorithms and computational methods
------------------------------------

Learn the mathematical methods and how to run them efficiently. These guides
use small demonstrations or API examples rather than a complete application.

.. container:: homepage-section-links

   .. toctree::
      :maxdepth: 1
      :caption: Algorithms and computational methods

      notebooks/linear_algebra_basics
      notebooks/svd_ht
      notebooks/dmd_intro
      notebooks/hodmd
      snapshot_statistics
      dft_pdft
      distributed_svd_spod

Worked examples
---------------

Follow a complete workflow, from loading or converting data to analysis and
visualization. Examples identify their input datasets; some require external
data. The PLT/SPOD example also includes a downloadable script.

.. container:: homepage-section-links

   .. toctree::
      :maxdepth: 1
      :caption: Worked examples

      plt_spod_example
      notebooks/svd_cylinder
      notebooks/dmd_cylinder
      notebooks/mssa_cylinder
      notebooks/dmd_naca0012_surface
      notebooks/piv_analysis
      notebooks/ipsp_explorer

.. toctree::
   :maxdepth: 1
   :caption: Python API

   flowtorch

.. toctree::
   :maxdepth: 1
   :caption: Releases

   release_notes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
