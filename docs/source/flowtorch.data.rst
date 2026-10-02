flowtorch.data 
==============

flowtorch.data.dataloader
-------------------------

.. automodule:: flowtorch.data.dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.foam\_dataloader
-------------------------------

.. automodule:: flowtorch.data.foam_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.foam\_numpy\_dataloader
--------------------------------------

.. automodule:: flowtorch.data.foam_numpy_dataloader
   :members:
   :undoc-members:
   :show-inheritance:


flowtorch.data.hdf5\_file
-------------------------

.. automodule:: flowtorch.data.hdf5_file
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.image\_dataloader
--------------------------------

.. automodule:: flowtorch.data.image_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.psp\_dataloader
------------------------------

.. automodule:: flowtorch.data.psp_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.vtk\_dataloader
------------------------------

.. automodule:: flowtorch.data.vtk_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.csv\_dataloader
------------------------------

.. automodule:: flowtorch.data.csv_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.tau\_dataloader
------------------------------

.. automodule:: flowtorch.data.tau_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.tecplot\_dataloader
----------------------------------

Select the field association when constructing the loader. The default
``association="point"`` reads vertex-centered data; ``association="cell"``
reads cell-centered data. ``field_names``, ``load_snapshot()``, ``locations``,
and ``weights`` follow this selection. ``vertices`` always returns mesh points.
An explicit association passed to ``load_snapshot()`` must match the constructor.

``weights`` contains raw cell lengths, areas, or volumes for cell data. For
point data, each cell contributes its measure divided by its vertex count to
each vertex. VTK computes cell measures and bulk NumPy operations accumulate
point contributions inside the isolated ParaView worker. Weights preserve
field ordering, use the configured dtype, and are cached per zone using the
first snapshot's geometry. This assumes a fixed mesh. The weights are neither
normalized nor square-rooted.

Mixed element types of the same dimension are supported. Mixed dimensions,
nonpositive or unsupported cell measures, isolated points, and ghost cells
raise errors; remove ghost cells before constructing the integration domain.
Equal sharing is a diagonal quadrature approximation for general elements,
and corresponds to mass lumping for linear simplices.

Some legacy TDV71 files expose ambiguous element types through the ParaView
reader. In the supplied NACA surface example, quads are reported as tetrahedra.
For a known element type, pass an explicit override rather than inferring it
from nearly zero volumes::

    loader = TecplotDataloader.from_tau(
        path, "alfa16.surface.pval.unsteady_",
        association="point", cell_type="quad",
    )
    pressure = loader.load_snapshot("pressure", loader.write_times[0])
    inner_product = (loader.weights * pressure.conj() * pressure).sum()

``cell_type`` accepts ``line``, ``triangle``, ``quad``, ``tetra``, or
``hexahedron``. It applies to geometric weights and cell locations, requires
matching connectivity sizes, and should only be used when the element type
is known. Normally leave it unset to retain the reader's topology.

.. automodule:: flowtorch.data.tecplot_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.selection\_tools
-------------------------------

.. automodule:: flowtorch.data.selection_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.masking\_tools
-----------------------------

.. automodule:: flowtorch.data.masking_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.interpolation\_tools
-----------------------------------

.. automodule:: flowtorch.data.interpolation_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.geometry\_tools
------------------------------

.. automodule:: flowtorch.data.geometry_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.differential\_tools
----------------------------------

.. automodule:: flowtorch.data.differential_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.utils
--------------------

.. automodule:: flowtorch.data.utils
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.outlier\_tools
-----------------------------

.. automodule:: flowtorch.data.outlier_tools
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.s\_cube\_dataloader
----------------------------------

.. automodule:: flowtorch.data.s_cube_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.sequence\_dataset
--------------------------------

.. automodule:: flowtorch.data.sequence_dataset
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.ensight\_dataloader
-------------------------------------

.. automodule:: flowtorch.data.ensight_dataloader
   :members:
   :undoc-members:
   :show-inheritance:

flowtorch.data.ensight\_writer
-------------------------------------

.. automodule:: flowtorch.data.ensight_writer
   :members:
   :undoc-members:
   :show-inheritance:
