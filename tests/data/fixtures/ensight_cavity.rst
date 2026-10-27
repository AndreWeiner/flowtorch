EnSight Gold fixtures
=====================

``ensight_cavity.zip`` contains OpenFOAM v2606-generated exports from the
GPL-licensed icoFoam lid-driven cavity tutorial: serial, dual polyhedral mesh,
and assembled two-process MPI runs. It includes ASCII/C-binary volume, patch,
cutting-plane and sampled-patch data, plus binary per-node data. Fields p and U
are computed; T and S are distinct constant tensor components for ordering tests.
Two snapshots are included, at 0.05 and 0.1 seconds. The archive compresses the
ASCII and binary files without changing their content.

Regenerate from the repository root with an activated OpenFOAM environment::

    simulation_setups/test_cases/ensight_cavity/Allrun
    python simulation_setups/test_cases/ensight_cavity/save_fixtures.py

Allrun requires MPI and either ParaView pvpython or Python plus VTK. Unit tests
extract this archive to a temporary directory; OpenFOAM and MPI are not required
for the tests. See the tutorial README.rst for geometry measures and provenance.
