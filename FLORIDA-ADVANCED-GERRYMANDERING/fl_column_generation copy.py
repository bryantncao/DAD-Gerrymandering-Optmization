"""
Florida congressional redistricting - column generation v4 (EG Maximization Focus)
"""

import os
import time
import random
import geopandas as gpd
import networkx as nx
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH SETUP & DATA LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print("Loading super-precinct data...", flush=True)

gdf = pd.read_pickle(os.path.join(OUTPUT_DIR, "fl_super_precincts.pkl"))
if not isinstance(gdf, gpd.GeoDataFrame):
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

gdf["UNIQUE_ID"] = gdf["super_precinct_id"].astype(str)
print(f"Total super-precincts loaded: {len(gdf)}", flush=True)

# ==========================================
# 2. ADJACENCY GRAPH & ISLAND CLEANUP
# ==========================================
print("Building adjacency graph and dropping disconnected islands...", flush=True)
adj_df = gdf[["UNIQUE_ID", "geometry"]].copy()
cand = gpd.sjoin(adj_df, adj_df, how="inner", predicate="intersects")
cand = cand[cand["UNIQUE_ID_left"] != cand["UNIQUE_ID_right"]].copy()
cand["pair"] = cand.apply(
    lambda r: tuple(sorted((r["UNIQUE_ID_left"], r["UNIQUE_ID_right"]))), axis=1
)
cand = cand.drop_duplicates(subset="pair")

geom_map = dict(zip(gdf["UNIQUE_ID"], gdf["geometry"]))
edge_rows = [
    (r["UNIQUE_ID_left"], r["UNIQUE_ID_right"])
    for _, r in cand.iterrows()
    if geom_map[r["UNIQUE_ID_left"]]
    .boundary.intersection(geom_map[r["UNIQUE_ID_right"]].boundary)
    .length > 0
]

G = nx.Graph()
for uid in gdf["UNIQUE_ID"]:
    G.add_node(uid)
for u, v in edge_rows:
    G.add_edge(u, v)

largest_cc = max(nx.connected_components(G), key=len)
dropped = set(G.nodes()) - set(largest_cc)
if dropped:
    print(f"  Dropping {len(dropped)} disconnected precincts.", flush=True)
    gdf = gdf[gdf["UNIQUE_ID"].isin(largest_cc)].copy()
    G = G.subgraph(largest_cc).copy()

precincts = list(gdf["UNIQUE_ID"])
n = len(precincts)
precincts_set = set(precincts)
pop = dict(zip(gdf["UNIQUE_ID"], gdf["population"]))
rep = dict(zip(gdf["UNIQUE_ID"], gdf["rep_votes"]))
dem = dict(zip(gdf["UNIQUE_ID"], gdf["dem_votes"]))
neighbors = {i: list(G.neighbors(i)) for i in precincts}

pol_rshare = {}
for i in precincts:
    t = rep[i] + dem[i]
    pol_rshare[i] = rep[i] / t if t > 0 else 0.5

# ==========================================
# 3. TARGETS
# ==========================================
p = 28
total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / p
tol = 0.45 
L = (1 - tol) * target_pop
U = (1 + tol) * target_pop

print(f"\n--- Problem Targets ---")
print(f"Precincts: {n}")
print(f"Districts: {p}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Target pop: {target_pop:,.0f} (allowed {L:,.0f} to {U:,.0f})")

max_precinct_pop = max(pop[i] for i in precincts)
if max_precinct_pop > U:
    print(f"  WARNING: largest single precinct has pop {max_precinct_pop:,}, which exceeds U = {U:,}.")

# ==========================================
# 4. CANDIDATE STATS HELPER
# ==========================================
def district_stats(members):
    members = frozenset(members)
    c_pop = sum(pop[i] for i in members)
    c_rep = sum(rep[i] for i in members)
    c_dem = sum(dem[i] for i in members)
    c_tot = c_rep + c_dem
    rwins = 1 if c_rep >= c_dem else 0
    
    if rwins:
        wr = c_rep - 0.5 * c_tot
        wd = c_dem
        # Malapportionment Bonus: If R wins, reward hitting the lower population limit
        pop_efficiency_bonus = (target_pop - c_pop) * 0.1 
    else:
        wr = c_rep
        wd = c_dem - 0.5 * c_tot
        # Malapportionment Bonus: If D wins, reward hitting the upper population limit
        pop_efficiency_bonus = (c_pop - target_pop) * 0.1 
        
    return {
        "members": members,
        "pop": c_pop,
        "rep": c_rep,
        "dem": c_dem,
        "rwins": rwins,
        "eg_term": (wd - wr) + pop_efficiency_bonus, # ADDED BONUS TO EG
    }

def partition_rwins(part):
    return sum(1 for d in part if sum(rep[i] for i in d) >= sum(dem[i] for i in d))

def partition_eg(part):
    tv = sum(rep[i] + dem[i] for i in precincts)
    WR = WD = 0.0
    for d in part:
        r = sum(rep[i] for i in d)
        dv = sum(dem[i] for i in d)
        t = r + dv
        if r >= dv:
            WR += r - 0.5 * t
            WD += dv
        else:
            WR += r
            WD += dv - 0.5 * t
    return (WD - WR) / tv if tv else 0.0

# ==========================================
# 5. FIXED COMPETITIVE BFS
# ==========================================
def get_bias_weight(v, bias):
    """Helper function to apply aggressive packing/cracking weights"""
    if bias == "prefer_R":
        # CRACK D: We want R to win, but efficiently. Avoid wasting strong R precincts.
        if pol_rshare[v] > 0.60:
            return 0.05
        return pol_rshare[v] + 0.1
    elif bias == "prefer_D":
        # PACK D: We want D to win by a massive landslide. Seek out the bluest precincts.
        return ((1.0 - pol_rshare[v]) ** 3) + 0.01
    else:
        return 1.0

def competitive_bfs_partition(rng, biases=None):
    seeds = rng.sample(precincts, p)
    if biases is None:
        biases = ["neutral"] * p

    assignment = {s: k for k, s in enumerate(seeds)}
    district_pop = [pop[s] for s in seeds]

    frontiers = [set() for _ in range(p)]
    for k, s in enumerate(seeds):
        for nb in neighbors[s]:
            if nb not in assignment:
                frontiers[k].add(nb)

    while len(assignment) < n:
        best_k = -1
        best_p = float("inf")
        for k in range(p):
            if frontiers[k] and district_pop[k] < best_p:
                best_p = district_pop[k]
                best_k = k
        if best_k == -1:
            return None 

        valid_picks = [v for v in frontiers[best_k] if v not in assignment]
        if not valid_picks:
            frontiers[best_k].clear()
            continue

        bias = biases[best_k]
        weights = [get_bias_weight(v, bias) for v in valid_picks]
        pick = rng.choices(valid_picks, weights=weights, k=1)[0]

        assignment[pick] = best_k
        district_pop[best_k] += pop[pick]
        for k in range(p):
            frontiers[k].discard(pick)
        for nb in neighbors[pick]:
            if nb not in assignment:
                frontiers[best_k].add(nb)

    if any(dp < L or dp > U for dp in district_pop):
        return None

    districts = [set() for _ in range(p)]
    for i, k in assignment.items():
        districts[k].add(i)
    return [frozenset(d) for d in districts]

# ==========================================
# 6. PARTITION VALIDATION
# ==========================================
def is_valid_partition(part):
    if len(part) != p:
        return False
    seen = set()
    for d in part:
        if len(d) == 0:
            return False
        if seen & d:
            return False
        seen |= d
        if not nx.is_connected(G.subgraph(d)):
            return False
        dp = sum(pop[i] for i in d)
        if dp < L or dp > U:
            return False
    return seen == precincts_set

# ==========================================
# 7. GENERATE PARTITIONS & HARVEST COLUMNS
# ==========================================
def generate_partitions(n_target, time_limit_sec, seed=42):
    rng = random.Random(seed)
    partitions = []
    harvested_districts = set()
    attempts = accepted = rejected_invalid = 0
    start = time.time()

    while len(partitions) < n_target:
        if time.time() - start > time_limit_sec:
            break
        attempts += 1

        biases = ["neutral"] * p if rng.random() < 0.2 else [rng.choices(["neutral", "prefer_R", "prefer_D"], weights=[0.2, 0.4, 0.4], k=1)[0] for _ in range(p)]
        part = competitive_bfs_partition(rng, biases=biases)
        if part is None:
            continue

        for d in part:
            dp = sum(pop[i] for i in d)
            if L <= dp <= U and nx.is_connected(G.subgraph(d)):
                harvested_districts.add(frozenset(d))

        if not is_valid_partition(part):
            rejected_invalid += 1
            continue

        partitions.append(part)
        accepted += 1

    elapsed = time.time() - start
    print(f"  Generated {accepted:,} valid partitions and HARVESTED {len(harvested_districts):,} individual valid districts in {elapsed:.1f}s.")
    return partitions, harvested_districts 


print("\n--- Phase 1: Generate valid partitions ---", flush=True)
N_PARTITIONS = 10000           # Increased limit
PART_TIME_LIMIT = 120         # TIME BUDGET: 10 Minutes
partitions, harvested_districts = generate_partitions(N_PARTITIONS, PART_TIME_LIMIT)

if not partitions:
    print("WARNING: Zero complete partitions generated. Proceeding to Gurobi using only harvested columns.")

OBJECTIVE_MODE = "eff_gap"   
if partitions:
    best_mc_part = max(partitions, key=partition_eg)
    best_mc_value = partition_eg(best_mc_part)
    print(f"  Monte Carlo best: EG = {best_mc_value:.4f}", flush=True)
else:
    best_mc_part = None
    best_mc_value = -float('inf')

# ==========================================
# 8. BUILD POOL & RUN MASTER IP
# ==========================================
def grow_random_candidate(seed, rng, bias="neutral", early_stop_prob=0.08):
    visited = {seed}
    cur_pop = pop[seed]
    frontier = set(neighbors[seed])
    while frontier:
        valid = [v for v in frontier if cur_pop + pop[v] <= U]
        if not valid:
            break
        if cur_pop >= L and rng.random() < early_stop_prob:
            break
            
        weights = [get_bias_weight(v, bias) for v in valid]
        pick = rng.choices(valid, weights=weights, k=1)[0]
        
        visited.add(pick)
        cur_pop += pop[pick]
        frontier.discard(pick)
        frontier.update(v for v in neighbors[pick] if v not in visited)
    if L <= cur_pop <= U:
        return frozenset(visited)
    return None

print("\n--- Phase 2: Build pool and run master IP ---", flush=True)
seen = set()
pool_members = []
for part in partitions:
    for d in part:
        if d not in seen:
            seen.add(d)
            pool_members.append(d)

for d in harvested_districts:
    if d not in seen:
        seen.add(d)
        pool_members.append(d)

print(f"  Initial pool size before diversifiers: {len(pool_members):,}")

N_EXTRA = 2000             # Increased limit
EXTRA_TIME_LIMIT = 300      # TIME BUDGET: 10 Minutes
rng_extra = random.Random(99)
t0 = time.time()
n_added = 0
while n_added < N_EXTRA and time.time() - t0 < EXTRA_TIME_LIMIT:
    s = rng_extra.choice(precincts)
    b = rng_extra.choices(
        ["neutral", "prefer_R", "prefer_D"], weights=[0.2, 0.4, 0.4], k=1
    )[0]
    sub = grow_random_candidate(s, rng_extra, bias=b)
    if sub is not None and sub not in seen:
        seen.add(sub)
        pool_members.append(sub)
        n_added += 1
        
print(f"  Pool size: {len(pool_members):,} (partitions anchors + {n_added:,} diversifiers)", flush=True)

pool = []
for k, members in enumerate(pool_members):
    s = district_stats(members)
    s["id"] = k
    pool.append(s)

covers = {i: [] for i in precincts}
for c in pool:
    for i in c["members"]:
        covers[i].append(c["id"])

best_ip_part = None
best_ip_value = None

try:
    options = {
        "WLSACCESSID": "",
        "WLSSECRET": "",
        "LICENSEID":,
    }
    env = gp.Env(params=options)
    m = gp.Model("set_partitioning_v4", env=env)

    z = m.addVars(len(pool), vtype=GRB.BINARY, name="z")
    for i in precincts:
        m.addConstr(
            gp.quicksum(z[cid] for cid in covers[i]) == 1, name=f"cover[{i}]"
        )
    m.addConstr(gp.quicksum(z[cid] for cid in range(len(pool))) == p, name="num_districts")

    if best_mc_part:
        best_mc_set = set(best_mc_part)
        mc_ids = [c["id"] for c in pool if c["members"] in best_mc_set]
        if len(mc_ids) == p:
            for cid in range(len(pool)):
                z[cid].Start = 1.0 if cid in set(mc_ids) else 0.0

    m.setObjective(
        gp.quicksum(pool[cid]["eg_term"] * z[cid] for cid in range(len(pool))),
        GRB.MAXIMIZE,
    )

    m.Params.OutputFlag = 1
    m.Params.TimeLimit = 300      # TIME BUDGET: 5 Minutes
    m.Params.MIPFocus = 1
    m.Params.Threads = 22
    m.Params.MIPGap = 0.01        # Allow early exit if 99% optimal
    m.Params.Presolve = 2

    print("  Solving master IP...", flush=True)
    t0 = time.time()
    m.optimize()
    print(f"  Master IP finished in {time.time()-t0:.1f}s (status={m.status}, SolCount={m.SolCount})", flush=True)

    if m.SolCount > 0:
        sel = [pool[cid] for cid in range(len(pool)) if z[cid].X > 0.5]
        if len(sel) == p and sum(len(c["members"]) for c in sel) == n:
            best_ip_part = [c["members"] for c in sel]
            best_ip_value = partition_eg(best_ip_part)
            print(f"  IP result: {best_ip_value} (vs Monte Carlo {best_mc_value})", flush=True)
except Exception as e:
    print(f"  Master IP error — falling back to Monte Carlo. ({e})", flush=True)

# ==========================================
# 9. PICK THE BETTER PLAN & REPORT
# ==========================================
if best_ip_value is not None and best_ip_value > best_mc_value:
    final_partition = best_ip_part
    source = "master IP"
else:
    final_partition = best_mc_part
    source = "Monte Carlo"

print(f"\nFinal plan source: {source}", flush=True)

selected = [district_stats(d) for d in final_partition]
assignment = {}
for k, c in enumerate(selected):
    for i in c["members"]:
        assignment[i] = k
gdf["final_district"] = gdf["UNIQUE_ID"].map(assignment)

total_rep_votes = sum(rep[i] for i in precincts)
total_dem_votes = sum(dem[i] for i in precincts)
total_votes = total_rep_votes + total_dem_votes

rows = []
total_R_wasted = 0.0
total_D_wasted = 0.0
for k, c in enumerate(selected):
    d_T = c["rep"] + c["dem"]
    r_share = c["rep"] / d_T if d_T else 0.0
    d_share = c["dem"] / d_T if d_T else 0.0
    winner = "R" if c["rwins"] else "D"
    if c["rwins"]:
        wr = c["rep"] - 0.5 * d_T
        wd = c["dem"]
    else:
        wr = c["rep"]
        wd = c["dem"] - 0.5 * d_T
    total_R_wasted += wr
    total_D_wasted += wd
    comps = nx.number_connected_components(G.subgraph(c["members"]))
    rows.append({
        "district": k, "winner": winner, "size": len(c["members"]),
        "population": c["pop"], "rep_votes": c["rep"], "dem_votes": c["dem"],
        "rep_share": r_share, "dem_share": d_share,
        "margin": r_share - d_share, "components": comps,
    })

df = pd.DataFrame(rows).sort_values("district").reset_index(drop=True)

print("\n----- District Summary Table -----", flush=True)
print(
    df[[
        "district", "winner", "size", "population",
        "rep_votes", "dem_votes", "rep_share", "dem_share",
        "margin", "components"
    ]].to_string(index=False),
    flush=True,
)

if (df["components"] != 1).any():
    print("\nWARNING: one or more districts is non-contiguous.", flush=True)

rep_seats = int((df["winner"] == "R").sum())
dem_seats = int((df["winner"] == "D").sum())
swide_R = total_rep_votes / total_votes if total_votes else 0.0
eg = (total_D_wasted - total_R_wasted) / total_votes if total_votes else None

print("\n----- Statewide Summary -----", flush=True)
print(f"Republican statewide vote share: {swide_R:.3f}")
print(f"Republican seat share: {rep_seats/p:.3f} ({rep_seats}/{p})")
print(f"\nEfficiency gap: {eg:.4f}" if eg is not None else "")

fig, ax = plt.subplots(figsize=(10, 12))
gdf.plot(column="final_district", categorical=True, cmap="tab20", edgecolor="black", linewidth=0.15, ax=ax)
title = f"{source} plan  |  R Seats: {rep_seats}/{p}"
if eg is not None:
    title += f"  |  EG: {eg:.4f}"
ax.set_title(title, fontweight="bold")
ax.set_axis_off()
plt.show()

gdf.to_pickle(os.path.join(OUTPUT_DIR, "fl_final_map_colgen_v4-fasttest.pkl"))
print(f"\nSaved to fl_final_map_colgen_v4.pkl")
