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

- **self**

### Python Example

```
from petsc4py import PETSc

dm = PETSc.DMPlex().createBoxMesh(
    faces=[4, 4],      # Number of elements per dimension
    lower=[0, 0],      # Bottom-left corner coordinates
    upper=[1, 1],      # Top-right corner coordinates
    simplex=False,     # True for triangles, False for Quadrilaterals
    interpolate=True   # Crucial: Creates edge/face entities for FEM
)
```

### Illustrative Example

...

### Notes

Here is the numbering returned for 2 faces in each direction for tensor cells:

```
 10---17---11---18----12
  |         |         |
  |         |         |
 20    2   22    3    24
  |         |         |
  |         |         |
  7---15----8---16----9
  |         |         |
  |         |         |
 19    0   21    1   23
  |         |         |
  |         |         |
  4---13----5---14----6
  ```

  and for simplicial cells:

  ```
   14----8---15----9----16
  |\     5  |\      7 |
  | \       | \       |
 13   2    14    3    15
  | 4   \   | 6   \   |
  |       \ |       \ |
 11----6---12----7----13
  |\        |\        |
  | \    1  | \     3 |
 10   0    11    1    12
  | 0   \   | 2   \   |
  |       \ |       \ |
  8----4----9----5----10
  ```