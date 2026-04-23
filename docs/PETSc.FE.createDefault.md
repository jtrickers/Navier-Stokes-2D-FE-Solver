# PETSc.FE().createDefault()

```
createDefault(dim, nc, isSimplex, qorder=DETERMINE, prefix=None, comm=None)
```

Create a FE for basic FEM computation.

This function defines the mathematical "Space" that the physics live in. For the Navier-Stokes equations, we use Taylor-Hood elements, which require a quadratic space for velocity and a linear space for pressure to satisfy the stability condition.

### Parameters

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

### Returns

- **self**