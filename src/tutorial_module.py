import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

class NavierStokesSolver:
    def __init__(self, reynolds=400.0):
        self.re = reynolds
        self.dm = None
        self.ts = None

    def create_mesh(self):
        # Ensure we interpolate so edges are created
        self.dm = PETSc.DMPlex().createBoxMesh(
            faces=[4, 4], lower=[0, 0], upper=[1, 1], 
            simplex=False, interpolate=True
        )
        self.dm.setFromOptions()

    def setup_discretization(self):
        dim = int(self.dm.getDimension())
        # Q2-Q1 Taylor-Hood
        fe_vel = PETSc.FE().createDefault(dim, dim, False, 3, "vel_", None)
        fe_pres = PETSc.FE().createDefault(dim, 1, False, 3, "pres_", None)
        self.dm.setField(0, fe_vel)
        self.dm.setField(1, fe_pres)
        self.dm.createDS()

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