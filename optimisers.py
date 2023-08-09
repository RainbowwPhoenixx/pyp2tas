import server
from typing import Union, List

from sko.PSO import PSO
from sko.tools import set_run_mode
from tqdm import tqdm


def pso_optimise(measure: server.MeasureFn, bounds: List[List[float]], pop: int, iters: int, init: Union[str, None] = None):
    global pbar, best_score

    print(f"Optimising with population size {pop} for {iters} iters.")

    servers = [
        ("127.0.0.1", 6555),
        ("127.0.0.1", 6556),
        ("127.0.0.1", 6557),
        ("127.0.0.1", 6558),
        ("127.0.0.1", 6559),
    ]
    pool = server.ServerPool(measure, servers, init)

    set_run_mode(pool.process_brute, "vectorization")

    pbar = tqdm(total=pop*iters)
    best_score = float('inf')

    pso = PSO(
        func=pool.process_brute,
        n_dim=len(bounds[0]),
        pop=pop,
        max_iter=iters-1,
        lb=bounds[0],  # type: ignore
        ub=bounds[1],  # type: ignore
        w=0.8, c1=0.5, c2=0.5
    )
    try:
        pso.run()
    except:
        print("Shit crap fuck! We had an unrecoverable error!")
        print("Here's the best we could do though:")

    best_pos, best_cost = pso.gbest_x, pso.gbest_y
    print(best_cost, best_pos)
