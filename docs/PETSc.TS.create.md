# PETSc.TS().create()

```
create(comm=None)
```

Create an empty timestepper

This function creates an emptry timestepper, after which the problem type can be set with `setProblemType` and the type of solver can be set with `setType`.

The TS (timestepper) object is the top-level solver for Ordinary Differential Equations (ODEs). Navier-Stokes is time-dependent; the TS object manages the evolution of the flow from $t_n$ to $t_{n+1}$.

### Parameters

- **comm : Comm | None**

    The MPI communicator, defaults to Sys.getDefaultComm.

### Returns

- **self**