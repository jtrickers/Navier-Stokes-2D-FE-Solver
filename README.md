# MA402 Final Project: Translation and Documentation of PETSc 

This code module contains code for a Navier-Stokes solver, translated from the original PETSc in C to Python. This code uses the time-dependent incompressible Navier-Stoked equation to solve for the flow of an incompressible fluid at low to moderate Reynolds number.

### From original code:

"Time dependent Navier-Stokes problem in 2d and 3d with finite elements. We solve the Navier-Stokes in a rectangular domain, using a parallel unstructured mesh (DMPLEX) to discretize it. This example supports discretized auxiliary fields (Re) as well as multilevel nonlinear solvers."

## PETSc.TS().create()

### petsc4py Reference [https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.TS.html#petsc4py.PETSc.TS.create]

create(comm=None)
Create an empty TS.

Collective.

The problem type can then be set with setProblemType and the type of solver can then be set with setType.

Parameters
:
comm (Comm | None) – MPI communicator, defaults to Sys.getDefaultComm.
Return type
:
Self
See also
TSCreate
Source code at petsc4py/PETSc/TS.pyx:220

### petsc4py Documentation [https://github.com/erdc/petsc4py/blob/master/src/PETSc/DMPlex.pyx]

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscTS newts = NULL
        CHKERR( TSCreate(ccomm, &newts) )
        PetscCLEAR(self.obj); self.ts = newts
        return self

### C Source (GitLab) [https://petsc.org/main/manualpages/TS/TSCreate/]

## PETSc.FE().createDefault()

### petsc4py Reference [https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.FE.html#petsc4py.PETSc.FE.createDefault]

createDefault(dim, nc, isSimplex, qorder=DETERMINE, prefix=None, comm=None)
Create a FE for basic FEM computation.

Collective.

Parameters
:
dim (int) – The spatial dimension.
nc (int) – The number of components.
isSimplex (bool) – Flag for simplex reference cell, otherwise it’s a tensor product.
qorder (int) – The quadrature order or DETERMINE to use Space polynomial degree.
prefix (str | None) – The options prefix, or None.
comm (Comm | None) – MPI communicator, defaults to Sys.getDefaultComm.
Return type
:
Self
See also
PetscFECreateDefault
Source code at petsc4py/PETSc/FE.pyx:76

## PETSc.DMPlex().createBoxMesh()

### petsc4py Reference [https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.DMPlex.html#petsc4py.PETSc.DMPlex.createBoxMesh]

createBoxMesh(faces, lower=(0, 0, 0), upper=(1, 1, 1), simplex=True, periodic=False, interpolate=True, localizationHeight=0, sparseLocalize=True, comm=None)
Create a mesh on the tensor product of intervals.

Collective.

Parameters
:
faces (Sequence[int]) – Number of faces per dimension, or None for the default.
lower (Sequence[float] | None) – The lower left corner.
upper (Sequence[float] | None) – The upper right corner.
simplex (bool | None) – True for simplices, False for tensor cells.
periodic (Sequence | str | int | bool | None) – The boundary type for the X, Y, Z direction, or None for DM.BoundaryType.NONE.
interpolate (bool | None) – Flag to create intermediate mesh entities (edges, faces).
localizationHeight (int | None) – Flag to localize edges and faces in addition to cells; only significant for periodic meshes.
sparseLocalize (bool | None) – Flag to localize coordinates only for cells near the periodic boundary; only significant for periodic meshes.
comm (Comm | None) – MPI communicator, defaults to Sys.getDefaultComm.
Return type
:
Self
See also
DM, DMPlex, DM.setFromOptions, DMPlex.createFromFile, DM.setType, DM.create, DMPlexCreateBoxMesh
Source code at petsc4py/PETSc/DMPlex.pyx:90

### petsc4py Documentation [https://github.com/erdc/petsc4py/blob/master/src/PETSc/TS.pyx]

*-no createBoxMesh, used create

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMPlexCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self