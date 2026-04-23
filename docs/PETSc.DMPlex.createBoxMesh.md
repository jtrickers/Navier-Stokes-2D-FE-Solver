# PETSc.TS().create()

```
createBoxMesh(faces, lower=(0, 0, 0), upper=(1, 1, 1), simplex=True, periodic=False, interpolate=True, localizationHeight=0, sparseLocalize=True, comm=None)
```

Create a mesh on the tensor product of intervals.

In computational fluid dynamics (CFD), the first step is defining the domain geometry. createBoxMesh generates an unstructured mesh (DMPlex) representing a rectangular (2D) or hexahedral (3D) domain. Unlike simple structured grids, a DMPlex maintains a complete "topology" (relationships between cells, faces, edges, and vertices), which is required for Finite Element Methods.

### Parameters

- **faces : Sequence[int]**

    Number of faces per dimension, or None for the default.

- **lower : Sequence[float] | None**

    The lower left corner.

- **upper : Sequence[float] | None**

    The upper right corner.
- **simplex : bool | None**

    True for simplices, False for tensor cells.

- **periodic : Sequence | str | int | bool | None**

    The boundary type for the X, Y, Z direction, or None for DM.BoundaryType.NONE.

- **interpolate : bool | None**

    Flag to create intermediate mesh entities (edges, faces).

- **localizationHeight : int | None**

    Flag to localize edges and faces in addition to cells; only significant for periodic meshes.

- **sparseLocalize : bool | None**

    Flag to localize coordinates only for cells near the periodic boundary; only significant for periodic meshes.

- **comm : Comm | None**

    MPI communicator, defaults to Sys.getDefaultComm.

### Returns

**self**