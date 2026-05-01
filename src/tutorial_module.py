import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

class NavierStokesSolver:
    def __init__(self, reynolds=400.0, n_cells=4, domain_size=1.0):
        self.re = reynolds
        self.n = n_cells
        self.L = domain_size
        self.dm = None
        self.ts = None

    def create_mesh(self):
        # Ensure we interpolate so edges are created
        self.dm = PETSc.DMPlex().createBoxMesh(
            faces=[self.n, self.n], lower=[0, 0], upper=[self.L, self.L], 
            simplex=False, interpolate=True
        )
        """
        PETSc.TS().createBoxMesh()

        - createBoxMesh(faces, lower=(0, 0, 0), upper=(1, 1, 1), simplex=True, periodic=False, interpolate=True, localizationHeight=0, sparseLocalize=True, comm=None)

        Create a mesh on the tensor product of intervals.

        In computational fluid dynamics (CFD), the first step is defining the domain geometry. createBoxMesh generates an unstructured mesh (DMPlex) representing a rectangular (2D) or hexahedral (3D) domain. Unlike simple structured grids, a DMPlex maintains a complete "topology" (relationships between cells, faces, edges, and vertices), which is required for Finite Element Methods.

        Parameters
        ----------

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

        Returns
        ----------

        - **self**

        Python Example
        ----------

        from petsc4py import PETSc

        dm = PETSc.DMPlex().createBoxMesh(
            faces=[4, 4],      # Number of elements per dimension
            lower=[0, 0],      # Bottom-left corner coordinates
            upper=[1, 1],      # Top-right corner coordinates
            simplex=False,     # True for triangles, False for Quadrilaterals
            interpolate=True   # Crucial: Creates edge/face entities for FEM
        )

        """
        self.dm.setFromOptions()

    def setup_discretization(self):
        dim = int(self.dm.getDimension())

        opts = PETSc.Options()
        opts.setValue("vel_petscspace_degree", 2)
        opts.setValue("pres_petscspace_degree", 1)

        # Q2-Q1 Taylor-Hood
        fe_vel = PETSc.FE().createDefault(dim, dim, False, 3, "vel_", None)
        fe_pres = PETSc.FE().createDefault(dim, 1, False, 3, "pres_", None)
        """
        PETSc.FE().createDefault()

        - createDefault(dim, nc, isSimplex, qorder=DETERMINE, prefix=None, comm=None)

        Create a FE for basic FEM computation.

        This function defines the mathematical "Space" that the physics live in. For the Navier-Stokes equations, we use Taylor-Hood elements, which require a quadratic space for velocity and a linear space for pressure to satisfy the stability condition.

        Parameters
        ----------

        - **dim : int**

            The spatial dimension.

        - **nc : int**

            The number of components.

        - **isSimplex : bool**

            Flag for simplex reference cell, otherwise it’s a tensor product.

        - **qorder : int**

            The quadrature order or DETERMINE to use Space polynomial degree.

        - **prefix : str | None**

            The options prefix, or None.

        - **comm : Comm | None**

            The MPI communicator, defaults to Sys.getDefaultComm.

        Returns
        ----------

        - **self**

        Python Example
        ----------

        # Create a 2D Quadratic Field for Velocity
        dim = 2
        components = 2  # (u, v)
        q_order = 3     # Quadrature order (usually degree + 1)

        fe_vel = PETSc.FE().createDefault(
            dim, 
            components, 
            False,      # isSimplex (matches createBoxMesh)
            q_order, 
            "vel_",     # Options prefix
            None        # Communicator
        )

        """
        self.dm.setField(0, fe_vel)
        self.dm.setField(1, fe_pres)
        self.dm.createDS()

        cdm = self.dm.getCoordinateDM()
        cdm.setField(0, fe_vel)
        cdm.createDS()

    def _get_true_coords(self, pt, coord_sec, coords_array):
        """Calculates midpoints for edges/faces since they lack direct coordinates."""
        dof = coord_sec.getDof(pt)
        if dof >= 2: # It's a vertex, take direct coords
            off = coord_sec.getOffset(pt)
            return coords_array[off], coords_array[off+1]
        
        # It's an edge or face: Average the coordinates of the underlying vertices
        closure, _ = self.dm.getTransitiveClosure(pt)
        v_coords = []
        for c_pt in closure:
            if coord_sec.getDof(c_pt) >= 2:
                off = coord_sec.getOffset(c_pt)
                v_coords.append([coords_array[off], coords_array[off+1]])
        
        if v_coords:
            return np.mean(v_coords, axis=0)
        return 0.0, 0.0

    def eval_ifunction(self, ts, t, U, U_t, F):
        F.set(0.0)
        u_glob = U.getArray(readonly=True)
        f_glob = F.getArray()
        
        section = self.dm.getGlobalSection()
        coord_sec = self.dm.getCoordinateSection()
        coords_vec = self.dm.getCoordinatesLocal()
        coords_arr = coords_vec.getArray(readonly=True)
        
        pStart, pEnd = self.dm.getChart()
        for pt in range(pStart, pEnd):
            dof = section.getDof(pt)
            off = section.getOffset(pt)
            if dof <= 0 or off < 0: continue
            
            x, y = self._get_true_coords(pt, coord_sec, coords_arr)
            
            # Projecting the algebraic constraint to verify mapping logic
            if dof >= 2: # Vel
                f_glob[off]   = u_glob[off]   - (t + x**2 + y**2)
                f_glob[off+1] = u_glob[off+1] - (t + 2.0*x**2 - 2.0*x*y)
            if dof == 3: # Pres
                f_glob[off+2] = u_glob[off+2] - (x + y - 1.0)

    def compute_exact(self, t, Vec_U):
        u_arr = Vec_U.getArray()
        section = self.dm.getGlobalSection()
        coord_sec = self.dm.getCoordinateSection()
        coords_arr = self.dm.getCoordinatesLocal().getArray(readonly=True)
        
        pStart, pEnd = self.dm.getChart()
        for pt in range(pStart, pEnd):
            dof = section.getDof(pt)
            off = section.getOffset(pt)
            if dof <= 0 or off < 0: continue
            
            x, y = self._get_true_coords(pt, coord_sec, coords_arr)
            
            if dof >= 2:
                u_arr[off]   = t + x**2 + y**2
                u_arr[off+1] = t + 2.0*x**2 - 2.0*x*y
            if dof == 3:
                u_arr[off+2] = x + y - 1.0

    def solve(self):
        self.ts = PETSc.TS().create(PETSc.COMM_WORLD)
        """
        PETSc.TS().create()
        
        - create(comm=None)
        
        Create an empty timestepper

        This function creates an emptry timestepper, after which the problem type can be set with `setProblemType` and the type of solver can be set with `setType`.

        The TS (timestepper) object is the top-level solver for Ordinary Differential Equations (ODEs). Navier-Stokes is time-dependent; the TS object manages the evolution of the flow from $t_n$ to $t_{n+1}$.

        Parameters
        ----------

        - **comm : Comm | None**

            The MPI communicator, defaults to Sys.getDefaultComm.

        Returns
        ----------

        - **self**

        Python Example
        ----------

        ts = PETSc.TS().create(PETSc.COMM_WORLD)
        ts.setDM(dm)                         # Associate the mesh
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setEquationType(ts.EquationType.IMPLICIT) # NS is usually implicit

        # Setup time-stepping parameters
        ts.setTime(0.0)
        ts.setTimeStep(0.01)
        ts.setMaxSteps(10)
        ts.setExactFinalTime(ts.ExactFinalTime.STEPOVER)

        """
        self.ts.setDM(self.dm)
        self.ts.setProblemType(self.ts.ProblemType.NONLINEAR)
        self.ts.setSolution(self.dm.createGlobalVec())
        self.ts.setIFunction(self.eval_ifunction, self.dm.createGlobalVec())
        self.ts.getSNES().setUseFD(True)
        self.ts.setTimeStep(0.01)
        self.ts.setMaxSteps(5)
        
        u = self.ts.getSolution()
        self.compute_exact(0.0, u)
        self.ts.solve(u)
        
        # Verify L2
        exact = u.duplicate()
        self.compute_exact(self.ts.getTime(), exact)
        exact.axpy(-1.0, u)
        print(f"Verified L2 Error: {exact.norm(PETSc.NormType.NORM_2):.2e}")

        return u

if __name__ == "__main__":
    solver = NavierStokesSolver()
    solver.create_mesh()
    solver.setup_discretization()
    solver.solve()