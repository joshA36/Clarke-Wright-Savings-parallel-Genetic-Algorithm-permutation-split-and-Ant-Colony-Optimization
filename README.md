# Clarke-Wright-Savings-parallel-Genetic-Algorithm-permutation-split-and-Ant-Colony-Optimization
import math
import time
import random
import requests
import numpy as np
from copy import deepcopy

# ---------- CONFIG ----------
URL = "https://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n34-k5.vrp"
RANDOM_SEED = 42

# ---------- PARSING ----------
def read_vrp_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    coords, demands = {}, {}
    capacity, depot = None, None
    mode = None
    for raw in lines:
        line = raw.strip()
        if line == "" : continue
        up = line.upper()
        if up.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif up.startswith("NODE_COORD_SECTION"):
            mode = "NODE"
            continue
        elif up.startswith("DEMAND_SECTION"):
            mode = "DEMAND"
            continue
        elif up.startswith("DEPOT_SECTION"):
            mode = "DEPOT"
            continue
        elif up.startswith("EOF"):
            break
        elif mode == "NODE":
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[idx] = (x, y)
        elif mode == "DEMAND":
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                d = int(parts[1])
                demands[idx] = d
        elif mode == "DEPOT":
            try:
                val = int(line.split()[0])
                if val != -1:
                    depot = val
            except:
                pass
    return coords, demands, depot, capacity

def euclidean(a, b):
    return int(round(math.hypot(a[0]-b[0], a[1]-b[1])))

def build_distance_matrix(coords):
    n = max(coords.keys())
    D = np.zeros((n+1, n+1), dtype=int)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                D[i,j] = 0
            else:
                D[i,j] = euclidean(coords[i], coords[j])
    return D

# ---------- UTIL ----------
def route_cost(route, D):
    # route includes depot at start and end e.g., [depot, a, b, depot]
    dist = 0
    for i in range(len(route)-1):
        dist += D[route[i], route[i+1]]
    return dist

def total_cost(routes, D):
    return sum(route_cost(r, D) for r in routes)

def route_load(route, demands):
    # route contains depot at start and end
    return sum(demands[n] for n in route if n in demands)

# 2-opt local improvement (on a single route, excluding depot ends)
def two_opt(route, D):
    # route: [depot, v1, v2, ..., depot]
    best = route
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_cost(new_route, D) < route_cost(best, D):
                    best = new_route
                    improved = True
        # loop until no improvement
    return best

# ---------- 1) Clarke-Wright Savings (parallel) ----------
def clarke_wright(coords, demands, depot, capacity, D, do_2opt=True):
    # Initialize: one route per customer: depot->i->depot
    customers = [i for i in coords.keys() if i != depot]
    routes = {i: [depot, i, depot] for i in customers}
    loads = {i: demands[i] for i in customers}

    # Compute savings S_ij = d(depot,i) + d(depot,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i >= j: continue
            s = D[depot,i] + D[depot,j] - D[i,j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Merge routes by savings if feasible
    for s, i, j in savings:
        # find routes containing i and j
        ri = rj = None
        for key, r in list(routes.items()):
            if r[1] == i and r[-2] == i:
                # single customer route, treat normally
                pass
            if i in r[1:-1]:
                ri = key
            if j in r[1:-1]:
                rj = key
            if ri and rj:
                break
        if ri is None or rj is None or ri == rj:
            # maybe i at end or j at start etc.
            # better approach: find route ids whose inner nodes contain i/j
            pass
        # robust approach: iterate through routes and find which contain i and j
        ri = rj = None
        for key, r in routes.items():
            if i in r[1:-1]:
                ri = key
            if j in r[1:-1]:
                rj = key
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # check that i is at the end of its route (just before depot) and j at start (just after depot),
        # or vice versa, depending on orientation for parallel CW merging
        can_merge = False
        # option 1: route_i ... i | depot and depot | j ... route_j  -> merge route_i without last depot and route_j without first depot
        if route_i[-2] == i and route_j[1] == j:
            new_load = loads[ri] + loads[rj]
            if new_load <= capacity:
                new_route = route_i[:-1] + route_j[1:]
                can_merge = True
        # option 2: route_j ... j | depot and depot | i ... route_i
        elif route_j[-2] == j and route_i[1] == i:
            new_load = loads[ri] + loads[rj]
            if new_load <= capacity:
                new_route = route_j[:-1] + route_i[1:]
                can_merge = True
        else:
            can_merge = False

        if can_merge:
            # remove old routes and add new
            new_key = min(ri, rj)
            del routes[ri]
            del routes[rj]
            routes[new_key] = new_route
            loads[new_key] = loads.get(ri,0) + loads.get(rj,0)

    # final collect routes
    final_routes = list(routes.values())
    # optional 2-opt on each route
    if do_2opt:
        final_routes = [two_opt(r, D) for r in final_routes]

    return final_routes

# ---------- 2) Genetic Algorithm ----------
# We'll use permutation representation with a split heuristic that builds feasible routes greedily.
def split_routes_from_permutation(perm, demands, capacity, depot):
    routes = []
    current_route = [depot]
    load = 0
    for node in perm:
        d = demands[node]
        if load + d <= capacity:
            current_route.append(node)
            load += d
        else:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot, node]
            load = d
    current_route.append(depot)
    routes.append(current_route)
    return routes

def ga_solve(coords, demands, depot, capacity, D,
             pop_size=50, gens=200, cx_prob=0.8, mut_prob=0.2, tournament_k=3):
    random.seed(RANDOM_SEED)
    customers = [i for i in coords.keys() if i != depot]
    # initial population: random perms + greedy seeds
    population = []
    # greedy seed: sort by demand desc
    greedy_perm = sorted(customers, key=lambda x: -demands[x])
    population.append(greedy_perm[:])
    # some nearest-neighbor seeds
    for start in customers[:5]:
        perm = nearest_neighbor_perm(start, customers, D, depot)
        population.append(perm)
    while len(population) < pop_size:
        p = customers[:]
        random.shuffle(p)
        population.append(p)

    def fitness(perm):
        routes = split_routes_from_permutation(perm, demands, capacity, depot)
        cost = total_cost(routes, D)
        return cost

    def tournament_select(pop, fits):
        best = None
        for _ in range(tournament_k):
            i = random.randrange(len(pop))
            if best is None or fits[i] < fits[best]:
                best = i
        return deepcopy(pop[best])

    def order_crossover(a, b):
        # OX crossover for permutations
        n = len(a)
        i, j = sorted(random.sample(range(n), 2))
        hole = set(a[i:j+1])
        child = [None]*n
        child[i:j+1] = a[i:j+1]
        idx = (j+1) % n
        for k in range(n):
            v = b[(j+1+k) % n]
            if v not in hole:
                child[idx] = v
                idx = (idx+1) % n
        return child

    def swap_mutation(ind):
        a = ind[:]
        i, j = random.sample(range(len(a)), 2)
        a[i], a[j] = a[j], a[i]
        return a

    def repair(perm):
        # ensure it's a permutation (should be)
        seen = set()
        out = []
        for x in perm:
            if x not in seen:
                out.append(x); seen.add(x)
        missing = [c for c in customers if c not in seen]
        out.extend(missing)
        return out

    # nearest neighbor helper
    def nearest_neighbor_perm(start, nodes, D, depot):
        unvisited = set(nodes)
        perm = []
        current = start
        perm.append(current)
        unvisited.remove(current)
        while unvisited:
            nxt = min(unvisited, key=lambda x: D[current,x])
            perm.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return perm

    # GA main loop
    population = population[:pop_size]
    fits = [fitness(ind) for ind in population]
    best_idx = min(range(len(fits)), key=lambda i: fits[i])
    best_perm = deepcopy(population[best_idx])
    best_cost = fits[best_idx]

    for gen in range(gens):
        new_pop = []
        # elitism: keep best
        new_pop.append(deepcopy(best_perm))
        while len(new_pop) < pop_size:
            parent1 = tournament_select(population, fits)
            parent2 = tournament_select(population, fits)
            if random.random() < cx_prob:
                child = order_crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)
            if random.random() < mut_prob:
                child = swap_mutation(child)
            child = repair(child)
            new_pop.append(child)
        population = new_pop
        fits = [fitness(ind) for ind in population]
        gen_best_idx = min(range(len(fits)), key=lambda i: fits[i])
        if fits[gen_best_idx] < best_cost:
            best_cost = fits[gen_best_idx]
            best_perm = deepcopy(population[gen_best_idx])
        # optional: print progress
        if (gen+1) % 50 == 0 or gen == 0:
            print(f"GA gen {gen+1}/{gens} best_cost={best_cost}")
    best_routes = split_routes_from_permutation(best_perm, demands, capacity, depot)
    # local improve routes by 2-opt
    best_routes = [two_opt(r, D) for r in best_routes]
    return best_routes

# ---------- 3) Ant Colony Optimization (constructive) ----------
def aco_solve(coords, demands, depot, capacity, D,
              n_ants=50, iterations=200, alpha=1.0, beta=2.0, rho=0.1, q0=0.9):
    random.seed(RANDOM_SEED)
    customers = [i for i in coords.keys() if i != depot]
    n = len(customers)
    # initialize pheromone on edges (i,j) for all i!=j (use small positive)
    tau0 = 1.0 / (n * np.mean(D[depot, customers]))
    tau = np.ones((max(coords.keys())+1, max(coords.keys())+1), dtype=float) * tau0
    eta = np.zeros_like(tau)
    for i in range(len(eta)):
        for j in range(len(eta)):
            if i!=j and i in coords and j in coords:
                eta[i,j] = 1.0 / (D[i,j] + 1e-6)

    best_routes = None
    best_cost = float('inf')

    for it in range(iterations):
        all_solutions = []
        for ant in range(n_ants):
            unvisited = set(customers)
            routes = []
            while unvisited:
                route = [depot]
                load = 0
                current = depot
                # build one route greedily until no feasible next
                while True:
                    feasible = [j for j in unvisited if load + demands[j] <= capacity]
                    if not feasible:
                        break
                    # compute probabilities
                    probs = []
                    denom = 0.0
                    for j in feasible:
                        denom += (tau[current,j]**alpha) * (eta[current,j]**beta)
                    # if denom == 0: choose randomly
                    if denom == 0:
                        nxt = random.choice(feasible)
                    else:
                        if random.random() < q0:
                            # exploitation: choose max pheromone*heuristic
                            nxt = max(feasible, key=lambda j: (tau[current,j]**alpha) * (eta[current,j]**beta))
                        else:
                            # probabilistic selection
                            r = random.random()
                            s = 0.0
                            for j in feasible:
                                val = (tau[current,j]**alpha) * (eta[current,j]**beta) / denom
                                s += val
                                if r <= s:
                                    nxt = j
                                    break
                    route.append(nxt)
                    load += demands[nxt]
                    unvisited.remove(nxt)
                    current = nxt
                route.append(depot)
                routes.append(route)
            # local improve routes
            routes = [two_opt(r, D) for r in routes]
            cost = total_cost(routes, D)
            all_solutions.append((routes, cost))

        # find best ant this iteration
        iter_best_routes, iter_best_cost = min(all_solutions, key=lambda x: x[1])
        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_routes = deepcopy(iter_best_routes)

        # pheromone update: evaporate
        tau *= (1 - rho)
        # deposit: for best solution in this iteration (or global best) deposit 1/cost on edges
        deposit = 1.0 / (iter_best_cost + 1e-6)
        for route in iter_best_routes:
            for i in range(len(route)-1):
                a, b = route[i], route[i+1]
                tau[a,b] += deposit
                # optionally tau[b,a] too if undirected
                tau[b,a] += deposit

        if (it+1) % 50 == 0 or it == 0:
            print(f"ACO iter {it+1}/{iterations} iter_best_cost={iter_best_cost} global_best={best_cost}")

    return best_routes

# ---------- MAIN ----------
def main():
    print("Reading instance from URL:", URL)
    coords, demands, depot, capacity = read_vrp_from_url(URL)
    print(f"Parsed: nodes={len(coords)}, depot={depot}, capacity={capacity}")
    D = build_distance_matrix(coords)

    # Seed rngs
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1) Clarke-Wright
    t0 = time.time()
    cw_routes = clarke_wright(coords, demands, depot, capacity, D, do_2opt=True)
    t1 = time.time()
    cw_cost = total_cost(cw_routes, D)
    print("\nClarke-Wright result:")
    for i,r in enumerate(cw_routes,1):
        print(f" R{i}: {r} load={route_load(r, demands)} cost={route_cost(r, D)}")
    print(f"Total distance: {cw_cost} time={(t1-t0):.2f}s")

    # 2) Genetic Algorithm
    t0 = time.time()
    ga_routes = ga_solve(coords, demands, depot, capacity, D,
                         pop_size=80, gens=300, cx_prob=0.9, mut_prob=0.2)
    t1 = time.time()
    ga_cost = total_cost(ga_routes, D)
    print("\nGenetic Algorithm result:")
    for i,r in enumerate(ga_routes,1):
        print(f" R{i}: {r} load={route_load(r, demands)} cost={route_cost(r, D)}")
    print(f"Total distance: {ga_cost} time={(t1-t0):.2f}s")

    # 3) Ant Colony Optimization
    t0 = time.time()
    aco_routes = aco_solve(coords, demands, depot, capacity, D,
                           n_ants=40, iterations=200, alpha=1.0, beta=3.0, rho=0.1, q0=0.9)
    t1 = time.time()
    aco_cost = total_cost(aco_routes, D)
    print("\nAnt Colony Optimization result:")
    for i,r in enumerate(aco_routes,1):
        print(f" R{i}: {r} load={route_load(r, demands)} cost={route_cost(r, D)}")
    print(f"Total distance: {aco_cost} time={(t1-t0):.2f}s")

    # summary
    print("\nSummary:")
    print(f" Clarke-Wright: cost={cw_cost:.0f}")
    print(f" GA:            cost={ga_cost:.0f}")
    print(f" ACO:           cost={aco_cost:.0f}")

if __name__ == "__main__":
    main()
