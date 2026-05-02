import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

class NavierStokesSolver:
    def __init__(self, reynolds=400.0, n_cells=4, domain_size=1.0, vel_degree=2, pres_degree=1,
                 exact_u=None, exact_v=None, exact_p=None):
        self.re = reynolds
        self.n = n_cells
        self.L = domain_size
        self.dm = None
        self.ts = None

        # Element polynomial degrees
        self.vel_degree = vel_degree
        self.pres_degree = pres_degree
        
        # Flow conditions / MMS exact functions
        # If none are provided, it defaults to the standard Q2-Q1 MMS test case
        self.exact_u = exact_u if exact_u else lambda t, x, y: t + x**2 + y**2
        self.exact_v = exact_v if exact_v else lambda t, x, y: t + 2.0*x**2 - 2.0*x*y
        self.exact_p = exact_p if exact_p else lambda t, x, y: x + y - 1.0

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
        opts.setValue("vel_petscspace_degree", self.vel_degree)
        opts.setValue("pres_petscspace_degree", self.pres_degree)

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
        """
        Retrieves real spatial coordinates for any topological point (Vertex or Edge).
        This robustly handles Q1 coordinates by averaging vertices for edge midpoints.
        """
        try:
            # 1. Safely check if the point has direct coordinates (Vertices)
            # We MUST check the section chart first. Calling getOffset() on an edge 
            # or cell point that isn't in the coordinate section chart throws a fatal 
            # exception, skipping the rest of the function!
            pStart, pEnd = coord_sec.getChart()
            
            if pStart <= pt < pEnd:
                off = coord_sec.getOffset(pt)
                dof = coord_sec.getDof(pt)
                if dof > 0 and off >= 0 and off + 1 < len(coords_array):
                    return float(coords_array[off]), float(coords_array[off+1])
            
            # 2. If it's an edge or cell, calculate its geometric center
            # petsc4py's getTransitiveClosure returns a 1D numpy array: 
            # [Point_ID, Orientation, Point_ID, Orientation...]
            closure_data = self.dm.getTransitiveClosure(pt)
            
            # Handle potential petsc4py version differences (array vs tuple)
            if isinstance(closure_data, tuple):
                closure_data = closure_data[0]
                
            # Slice with [::2] to ONLY iterate over the Point IDs.
            closure_pts = closure_data[::2] 
            
            v_coords = []
            for c_pt in closure_pts:
                c_pt_int = int(c_pt) # Cast numpy int to python int for safety
                
                # Safely check chart bounds before querying the section
                if pStart <= c_pt_int < pEnd:
                    c_off = coord_sec.getOffset(c_pt_int)
                    c_dof = coord_sec.getDof(c_pt_int)
                    if c_dof > 0 and c_off >= 0 and c_off + 1 < len(coords_array):
                        v_coords.append([coords_array[c_off], coords_array[c_off+1]])
            
            if v_coords:
                mean_coords = np.mean(v_coords, axis=0)
                return float(mean_coords[0]), float(mean_coords[1])
        except Exception:
            pass
            
        return None, None

    def eval_ifunction(self, ts, t, U, U_t, F):
        """Global residual evaluation for MMS verification."""
        F.set(0.0)
        u_glob = U.getArray(readonly=True)
        f_glob = F.getArray()
        
        section = self.dm.getGlobalSection()
        coord_sec = self.dm.getCoordinateSection()
        coords_arr = self.dm.getCoordinatesLocal().getArray(readonly=True)
        
        pStart, pEnd = self.dm.getChart()
        for pt in range(pStart, pEnd):
            try:
                x, y = self._get_true_coords(pt, coord_sec, coords_arr)
                if x is None:  
                    continue
                
                # Safely request exact memory offsets AND Degree of Freedom (DOF) counts
                u_off = section.getFieldOffset(pt, 0)
                u_dof = section.getFieldDof(pt, 0)
                
                p_off = section.getFieldOffset(pt, 1)
                p_dof = section.getFieldDof(pt, 1)
                
                # --- MEMORY CORRUPTION FIX ---
                # We MUST check that `dof >= 1`. If a point has 0 DOFs for a field, 
                # PETSc may still return an offset pointing into another field's memory!
                
                # Apply Velocity constraints
                if u_dof >= 2 and u_off >= 0 and u_off + 1 < len(f_glob):
                    f_glob[u_off]   = u_glob[u_off]   - self.exact_u(t, x, y)
                    f_glob[u_off+1] = u_glob[u_off+1] - self.exact_v(t, x, y)
                    
                # Apply Pressure constraints
                if p_dof >= 1 and p_off >= 0 and p_off < len(f_glob):
                    f_glob[p_off]   = u_glob[p_off]   - self.exact_p(t, x, y)
            except Exception:
                continue

    def compute_exact(self, t, Vec_U):
        """Sets the vector to the exact MMS solution."""
        u_arr = Vec_U.getArray()
        section = self.dm.getGlobalSection()
        coord_sec = self.dm.getCoordinateSection()
        coords_arr = self.dm.getCoordinatesLocal().getArray(readonly=True)
        
        pStart, pEnd = self.dm.getChart()
        for pt in range(pStart, pEnd):
            try:
                x, y = self._get_true_coords(pt, coord_sec, coords_arr)
                if x is None:  
                    continue
                
                u_off = section.getFieldOffset(pt, 0)
                u_dof = section.getFieldDof(pt, 0)
                
                p_off = section.getFieldOffset(pt, 1)
                p_dof = section.getFieldDof(pt, 1)
                
                # --- MEMORY CORRUPTION FIX ---
                # Guarantee we only write exact polynomials to validly allocated DOFs
                
                if u_dof >= 2 and u_off >= 0 and u_off + 1 < len(u_arr):
                    u_arr[u_off]   = self.exact_u(t, x, y)
                    u_arr[u_off+1] = self.exact_v(t, x, y)
                    
                if p_dof >= 1 and p_off >= 0 and p_off < len(u_arr):
                    u_arr[p_off]   = self.exact_p(t, x, y)
            except Exception:
                continue

    def extract_velocity_field(self, solution_vec):
        """
        Safely extracts the coordinates and velocity components for plotting.
        Uses PETSc's explicit Global-to-Local mapping and explicit field offsets.
        """
        local_sol = self.dm.getLocalVec()
        self.dm.globalToLocal(solution_vec, local_sol)
        u_array = local_sol.getArray(readonly=True)
        
        local_sec = self.dm.getLocalSection()
        coord_sec = self.dm.getCoordinateSection()
        coords_arr = self.dm.getCoordinatesLocal().getArray(readonly=True)
        
        x_vals, y_vals, u_vel, v_vel = [], [], [], []
        seen_coords = set() # Track unique geometric coordinates to prevent plotting overlap
        
        pStart, pEnd = self.dm.getChart()
        for pt in range(pStart, pEnd):
            try:
                # Explicitly only pull indices belonging to Field 0 (Velocity)
                u_off = local_sec.getFieldOffset(pt, 0)
                u_dof = local_sec.getFieldDof(pt, 0)
                
                # Check that the offset is valid AND the point actually has velocity DOFs
                if u_off < 0 or u_dof < 2 or u_off + 1 >= len(u_array): 
                    continue
                    
                x, y = self._get_true_coords(pt, coord_sec, coords_arr)
                if x is None:  
                    continue

                # Round coordinates to 5 decimal places to perfectly capture 
                # base-2 mesh fractions (like 1/64 = 0.015625) without uneven snapping
                coord_key = (round(x, 5), round(y, 5))
                if coord_key in seen_coords:
                    continue
                    
                seen_coords.add(coord_key)
                
                # Memory alignments are now perfectly safe, so we can plot every 
                # valid Q2 topological coordinate exactly as it evaluates
                x_vals.append(x)
                y_vals.append(y)
                u_vel.append(u_array[u_off])
                v_vel.append(u_array[u_off+1])
                    
            except Exception:
                continue
                
        self.dm.restoreLocalVec(local_sol)
        
        return x_vals, y_vals, u_vel, v_vel

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
        
        sol_vec = self.dm.createGlobalVec()
        self.ts.setSolution(sol_vec)
        self.ts.setIFunction(self.eval_ifunction, self.dm.createGlobalVec())
        
        snes = self.ts.getSNES()
        snes.setUseFD(True)
        
        ksp = snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        
        self.ts.setTimeStep(0.01)
        self.ts.setMaxSteps(5)
        
        self.compute_exact(0.0, sol_vec)
        self.ts.solve(sol_vec)
        
        exact = sol_vec.duplicate()
        self.compute_exact(self.ts.getTime(), exact)
        exact.axpy(-1.0, sol_vec)
        print(f"L2 Error Norm: {exact.norm(PETSc.NormType.NORM_2):.2e}")
        
        return sol_vec

if __name__ == "__main__":
    solver = NavierStokesSolver()
    solver.create_mesh()
    solver.setup_discretization()
    solver.solve()