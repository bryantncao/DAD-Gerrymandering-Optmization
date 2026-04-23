"""
Florida 28-District Optimized Gerrymandering Solver

"""

import geopandas as gpd
import networkx as nx
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH SETUP & DATA LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

print("Loading super-precinct data...", flush=True)
gdf = pd.read_pickle(os.path.join(OUTPUT_DIR, "fl_super_precincts.pkl"))
if not isinstance(gdf, gpd.GeoDataFrame):
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

gdf["UNIQUE_ID"] = gdf["super_precinct_id"].astype(str)
print(f"Total super-precincts loaded: {len(gdf)}", flush=True)

# ==========================================
# 2. DYNAMIC ADJACENCY & GRAPH CLEANUP
# ==========================================
print("Calculating physical borders and dropping islands...", flush=True)
adj_df = gdf[["UNIQUE_ID", "geometry"]].copy()

cand = gpd.sjoin(adj_df, adj_df, how="inner", predicate="intersects")
cand = cand[cand["UNIQUE_ID_left"] != cand["UNIQUE_ID_right"]].copy()
cand["pair"] = cand.apply(
    lambda r: tuple(sorted((r["UNIQUE_ID_left"], r["UNIQUE_ID_right"]))),
    axis=1
)
cand = cand.drop_duplicates(subset="pair")

geom_map = dict(zip(gdf["UNIQUE_ID"], gdf["geometry"]))
edge_rows = [
    (r["UNIQUE_ID_left"], r["UNIQUE_ID_right"])
    for _, r in cand.iterrows()
    if geom_map[r["UNIQUE_ID_left"]].boundary.intersection(
        geom_map[r["UNIQUE_ID_right"]].boundary).length > 0
]
edges = pd.DataFrame(edge_rows, columns=["u", "v"])

G = nx.Graph()
for uid in gdf["UNIQUE_ID"]:
    G.add_node(uid)
for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

largest_cc = max(nx.connected_components(G), key=len)
dropped_nodes = set(G.nodes()) - set(largest_cc)

if dropped_nodes:
    print(f"⚠️  Dropping {len(dropped_nodes)} disconnected islands.", flush=True)
    gdf = gdf[gdf["UNIQUE_ID"].isin(largest_cc)].copy()
    edges = edges[edges["u"].isin(largest_cc) & edges["v"].isin(largest_cc)].copy()

G = nx.Graph()
for uid in gdf["UNIQUE_ID"]:
    G.add_node(uid)
for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

# ==========================================
# 3. OPTIMIZATION SETUP
# ==========================================
precincts = list(gdf["UNIQUE_ID"])
n = len(precincts)

# --- CONFIG ---
p = 28                      # Florida congressional districts
TOLERANCE = 0.15            # Population tolerance (15%)
MIP_GAP = 0.20              # Accept solutions within 20% of optimal
TIME_LIMIT = 3600 * 5       # 6 hours max
OBJECTIVE_MODE = "eff_gap"  # "rep_wins" or "eff_gap"
STOP_AT_FIRST = False       # Set True to stop at first feasible solution
# --------------

districts = list(range(p))

pop = dict(zip(gdf["UNIQUE_ID"], gdf["population"]))
rep = dict(zip(gdf["UNIQUE_ID"], gdf["rep_votes"]))
dem = dict(zip(gdf["UNIQUE_ID"], gdf["dem_votes"]))

total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / p
L = (1 - TOLERANCE) * target_pop
U = (1 + TOLERANCE) * target_pop

total_rep_votes = sum(rep[i] for i in precincts)
total_dem_votes = sum(dem[i] for i in precincts)
total_votes = total_rep_votes + total_dem_votes
Mvote = total_votes if total_votes > 0 else 1

print(f"\n--- Model Targets ---", flush=True)
print(f"Districts: {p}", flush=True)
print(f"Target pop: {target_pop:,.0f} (range {L:,.0f} to {U:,.0f})", flush=True)
print(f"Total R: {total_rep_votes:,} | Total D: {total_dem_votes:,}", flush=True)
print(f"R statewide share: {total_rep_votes/total_votes:.3f}", flush=True)

# ==========================================
# 4. BUILD GUROBI MODEL
# ==========================================
print("\nBuilding Gurobi model (no flow vars — using lazy cuts)...", flush=True)

options = {
    "WLSACCESSID": "",
    "WLSSECRET": "",
    "LICENSEID": # INSERT YOUR GUROBI LICENSE
}
env = gp.Env(params=options)
m = gp.Model("fl_fast_gerrymander", env=env)

# Variables: assignment, roots, winner, EG helper
x = m.addVars(precincts, districts, vtype=GRB.BINARY, name="x")
r = m.addVars(precincts, districts, vtype=GRB.BINARY, name="r")
rep_win = m.addVars(districts, vtype=GRB.BINARY, name="rep_win")
u_var = m.addVars(districts, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

# Each precinct assigned to exactly one district
m.addConstrs(
    (gp.quicksum(x[i, k] for k in districts) == 1 for i in precincts),
    name="assign"
)

# Exactly one root per district; root must be in that district
m.addConstrs(
    (gp.quicksum(r[i, k] for i in precincts) == 1 for k in districts),
    name="one_root"
)
m.addConstrs(
    (r[i, k] <= x[i, k] for i in precincts for k in districts),
    name="root_in_district"
)

# Symmetry breaking: districts ordered by strict root index
prec_order = {pid: idx for idx, pid in enumerate(precincts)}
root_idx = {
    k: gp.quicksum(prec_order[i] * r[i, k] for i in precincts)
    for k in districts
}
for k in range(p - 1):
    m.addConstr(root_idx[k] <= root_idx[k + 1] - 1,
                name=f"root_order_{k}")

# Population balance
m.addConstrs(
    (gp.quicksum(pop[i] * x[i, k] for i in precincts) >= L
     for k in districts),
    name="pop_lb"
)
m.addConstrs(
    (gp.quicksum(pop[i] * x[i, k] for i in precincts) <= U
     for k in districts),
    name="pop_ub"
)

# Vote expressions
Rk = {k: gp.quicksum(rep[i] * x[i, k] for i in precincts) for k in districts}
Dk = {k: gp.quicksum(dem[i] * x[i, k] for i in precincts) for k in districts}
Tk = {k: Rk[k] + Dk[k] for k in districts}

# Winner indicator and u_var linearization for efficiency gap
for k in districts:
    m.addConstr(Rk[k] - Dk[k] >= 1 - Mvote * (1 - rep_win[k]),
                name=f"rep_wins_{k}")
    m.addConstr(Dk[k] - Rk[k] >= 0 - Mvote * rep_win[k],
                name=f"dem_wins_{k}")
    # u_var[k] = rep_win[k] * Tk[k]
    m.addConstr(u_var[k] <= Mvote * rep_win[k])
    m.addConstr(u_var[k] <= Tk[k])
    m.addConstr(u_var[k] >= Tk[k] - Mvote * (1 - rep_win[k]))

WR = {k: Rk[k] - 0.5 * u_var[k] for k in districts}
WD = {k: Dk[k] - 0.5 * (Tk[k] - u_var[k]) for k in districts}

# Objective
if OBJECTIVE_MODE == "rep_wins":
    m.setObjective(gp.quicksum(rep_win[k] for k in districts), GRB.MAXIMIZE)
elif OBJECTIVE_MODE == "eff_gap":
    m.setObjective(gp.quicksum(WD[k] - WR[k] for k in districts), GRB.MAXIMIZE)
else:
    raise ValueError(f"Unknown OBJECTIVE_MODE: {OBJECTIVE_MODE}")

# ==========================================
# 5. LAZY-CUT CONTIGUITY CALLBACK
# ==========================================
def contiguity_callback(model, where):
    """When integer-feasible solution found, ensure each district is connected."""
    if where != GRB.Callback.MIPSOL:
        return
    xv = model.cbGetSolution(x)
    rv = model.cbGetSolution(r)

    for k in districts:
        members = [i for i in precincts if xv[i, k] > 0.5]
        if len(members) <= 1:
            continue
        H = G.subgraph(members)
        if nx.is_connected(H):
            continue

        # Identify anchor component (contains the root)
        root_node = next((i for i in precincts if rv[i, k] > 0.5), None)
        if root_node and root_node in H:
            anchor = nx.node_connected_component(H, root_node)
        else:
            anchor = max(nx.connected_components(H), key=len)

        # Cut off each disconnected component
        for comp in nx.connected_components(H):
            if comp & anchor:
                continue
            # Boundary: neighbors outside comp
            boundary = set()
            for node in comp:
                for nbr in G.neighbors(node):
                    if nbr not in comp:
                        boundary.add(nbr)
            if not boundary:
                continue
            # Separator cut: if all of comp is in district k, at least one
            # boundary node must also be in district k
            model.cbLazy(
                gp.quicksum(x[i, k] for i in comp)
                <= len(comp) * gp.quicksum(x[j, k] for j in boundary)
            )

# ==========================================
# 6. GREEDY WARM START
# ==========================================
def warm_start():
    """Region-growing from p farthest-apart seeds to provide a feasible start."""
    # Pick seeds by farthest-point iteration
    seeds = [max(precincts, key=lambda i: pop[i])]
    dists = dict(nx.single_source_shortest_path_length(G, seeds[0]))
    while len(seeds) < p:
        nxt = max(precincts, key=lambda i: dists.get(i, 0))
        if nxt in seeds:
            # Fall back to unused precinct
            remaining = [i for i in precincts if i not in seeds]
            if not remaining:
                break
            nxt = remaining[0]
        seeds.append(nxt)
        new_d = nx.single_source_shortest_path_length(G, nxt)
        for node in precincts:
            dists[node] = min(dists.get(node, 9999), new_d.get(node, 9999))

    assign = {s: k for k, s in enumerate(seeds)}
    dpop = {k: pop[seeds[k]] for k in districts}
    frontier = {
        k: set(G.neighbors(seeds[k])) - set(seeds)
        for k in districts
    }

    while len(assign) < n:
        avail_ks = [k for k in districts if (frontier[k] - set(assign))]
        if not avail_ks:
            # Assign stragglers to any adjacent district
            for i in precincts:
                if i in assign:
                    continue
                for nbr in G.neighbors(i):
                    if nbr in assign:
                        assign[i] = assign[nbr]
                        dpop[assign[nbr]] += pop[i]
                        break
                else:
                    assign[i] = 0
            break

        # Grow district with smallest current population
        k = min(avail_ks, key=lambda kk: dpop[kk])
        cands = list(frontier[k] - set(assign))
        if not cands:
            continue
        # Pick candidate closest to seed (keeps districts compact)
        pick = min(cands,
                   key=lambda c: nx.shortest_path_length(G, c, seeds[k]))
        assign[pick] = k
        dpop[k] += pop[pick]
        frontier[k].update(
            nbr for nbr in G.neighbors(pick) if nbr not in assign
        )

    # Write to Gurobi Start values
    for i in precincts:
        k_assigned = assign.get(i, 0)
        for kk in districts:
            x[i, kk].Start = 1.0 if kk == k_assigned else 0.0
    for kk, s in enumerate(seeds):
        for i in precincts:
            r[i, kk].Start = 1.0 if i == s else 0.0

    print(f"Warm start: pop range "
          f"{min(dpop.values()):,.0f} to {max(dpop.values()):,.0f} "
          f"(target: {target_pop:,.0f})", flush=True)

try:
    warm_start()
except Exception as e:
    print(f"⚠️  Warm start failed: {e}. Continuing without.", flush=True)

# ==========================================
# 7. SOLVER PARAMS (tuned for fast feasibility)
# ==========================================
m.Params.OutputFlag = 1
m.Params.TimeLimit = TIME_LIMIT
m.Params.MIPFocus = 1              # Prioritize feasibility
m.Params.Heuristics = 0.25         # Moderate heuristic effort
m.Params.Cuts = 1                  # Moderate cut generation
m.Params.Presolve = 2              # Aggressive presolve
m.Params.MIPGap = MIP_GAP          # Accept within 10% of optimal
m.Params.LazyConstraints = 1       # REQUIRED for cbLazy
m.Params.Threads = 0               # Use all available
m.Params.ImproveStartGap = 0.25    # Switch to improving solutions earlier

if STOP_AT_FIRST:
    m.Params.SolutionLimit = 1
    print("⚡ STOP_AT_FIRST enabled — will halt at first feasible map.", flush=True)

print("\n🚀 Starting optimization...\n", flush=True)
m.optimize(contiguity_callback)

# ==========================================
# 8. EXTRACT & REPORT SOLUTION
# ==========================================
if m.SolCount > 0:
    print("\n✅ Districting plan found!\n", flush=True)

    assignment = {}
    for i in precincts:
        for k in districts:
            if x[i, k].X > 0.5:
                assignment[i] = k
                break
    gdf["final_district"] = gdf["UNIQUE_ID"].map(assignment)

    # Verify contiguity of final solution
    print("Verifying contiguity of final districts...", flush=True)
    all_connected = True
    for k in districts:
        members = [i for i in precincts if assignment.get(i) == k]
        if len(members) > 1:
            H = G.subgraph(members)
            if not nx.is_connected(H):
                print(f"   ⚠️  District {k} is NOT connected!", flush=True)
                all_connected = False
    if all_connected:
        print("   ✅ All districts contiguous.", flush=True)

    # District summary
    total_rep_wasted = 0.0
    total_dem_wasted = 0.0
    district_summary = []

    for k in districts:
        members = [i for i in precincts if assignment.get(i) == k]
        dp = sum(pop[i] for i in members)
        dr = sum(rep[i] for i in members)
        dd = sum(dem[i] for i in members)
        dt = dr + dd
        rs = dr / dt if dt > 0 else 0.0
        ds = dd / dt if dt > 0 else 0.0
        winner = "R" if rep_win[k].X > 0.5 else "D"
        wr_val = WR[k].getValue()
        wd_val = WD[k].getValue()
        total_rep_wasted += wr_val
        total_dem_wasted += wd_val

        district_summary.append({
            "district": k, "winner": winner, "population": dp,
            "rep_votes": dr, "dem_votes": dd, "total_votes": dt,
            "rep_share": rs, "dem_share": ds, "margin": rs - ds,
            "rep_wasted": wr_val, "dem_wasted": wd_val
        })

    district_df = pd.DataFrame(district_summary).sort_values("district").reset_index(drop=True)

    print("\n----- District Summary -----", flush=True)
    print(district_df[[
        "district", "winner", "population", "rep_votes", "dem_votes",
        "rep_share", "dem_share", "margin"
    ]].to_string(index=False), flush=True)

    # Statewide summary
    statewide_rep_share = total_rep_votes / total_votes
    rep_seats = int((district_df["winner"] == "R").sum())
    dem_seats = p - rep_seats

    print(f"\n----- Statewide Summary -----", flush=True)
    print(f"R statewide vote share: {statewide_rep_share:.3f}", flush=True)
    print(f"R seat share: {rep_seats/p:.3f} ({rep_seats} / {p})", flush=True)
    print(f"D seat share: {dem_seats/p:.3f} ({dem_seats} / {p})", flush=True)

    eg_numerator = total_dem_wasted - total_rep_wasted
    eg = eg_numerator / total_votes
    print(f"\n----- Efficiency Gap -----", flush=True)
    print(f"R wasted: {total_rep_wasted:,.0f}", flush=True)
    print(f"D wasted: {total_dem_wasted:,.0f}", flush=True)
    print(f"Efficiency gap: {eg:.4f}", flush=True)

    # Packing/cracking diagnostics
    packed_thresh = 0.65
    packed_D = district_df[(district_df["winner"] == "D")
                            & (district_df["dem_share"] >= packed_thresh)]
    cracked_D = district_df[(district_df["winner"] == "R")
                             & (district_df["dem_share"].between(0.40, 0.49))]

    print(f"\n----- Packing/Cracking Flags -----", flush=True)
    if len(packed_D) > 0:
        print(f"Packed D districts ({len(packed_D)}):", flush=True)
        print(packed_D[["district", "dem_share", "margin"]].to_string(index=False), flush=True)
    if len(cracked_D) > 0:
        print(f"Cracked D districts ({len(cracked_D)}):", flush=True)
        print(cracked_D[["district", "dem_share", "margin"]].to_string(index=False), flush=True)

    # Save and plot
    gdf.to_pickle(os.path.join(OUTPUT_DIR, "fl_final_map.pkl"))
    print("\nMap saved to fl_final_map.pkl", flush=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(column="final_district", categorical=True, cmap="tab20",
             edgecolor="black", linewidth=0.2, ax=ax)
    ax.set_title(f"FL {p} Districts — R seats: {rep_seats}/{p} | "
                 f"EG: {eg:.3f}", fontweight='bold')
    ax.set_axis_off()
    plt.show()
else:
    print(f"\n❌ No solution found. Status: {m.status}", flush=True)
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible. Try:", flush=True)
        print("  - Increasing TOLERANCE (e.g., 0.10)", flush=True)
        print("  - Reducing TARGET_SUPER_PRECINCTS in combine_precincts.py", flush=True)
        print("  - Checking super-precinct connectivity", flush=True)
