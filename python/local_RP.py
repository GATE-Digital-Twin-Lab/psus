import numpy as np

def RP111(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 2
    beta = 5

    if x.shape[1] != nrv_e:
        raise ValueError(f"RP111 expects {nrv_e} variables, got {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    g = (beta**2) / 2 - np.abs(x1 * x2)

    if g.size == 1:
        return g.item()   
    return g

def RP14(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 5
    if x.shape[1] != nrv_e:
        raise ValueError(f"RP14 expects {nrv_e} variables, got {x.shape[1]}")

    g = (
        x[:, 0]
        - 32 / (np.pi * x[:, 1] ** 3)
        * np.sqrt((x[:, 2] ** 2 * x[:, 3] ** 2) / 16 + x[:, 4] ** 2)
    )


    if g.size == 1:
        return g.item()   
    return g

def RP22(x):

    x = np.atleast_2d(np.array(x, dtype=float))

    nrv_e = 2
    if x.shape[1] != nrv_e:
        raise ValueError(f"RP22 expects {nrv_e} variables, got {x.shape[1]}")

    x1 = x[:, 0]
    x2 = x[:, 1]

    g = (
        2.5
        - (1 / np.sqrt(2)) * (x1 + x2)
        + 0.1 * (x1 - x2) ** 2
    )
    if g.size == 1:
        return g.item()   
    return g
