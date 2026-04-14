

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import wkb
import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------
# 1. Read Nevada precinct/VTD shapefile
# --------------------------------------------------
gdf = gpd.read_file(r"C:\Users\benne\Downloads\nv_2020\nv_2020.shp")

# --------------------------------------------------
# 2. Drop Z dimension if needed
# --------------------------------------------------
gdf["geometry"] = gdf["geometry"].apply(
    lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))
)

# --------------------------------------------------
# 3. Make geometries valid
# --------------------------------------------------
try:
    gdf["geometry"] = gdf.geometry.make_valid()
except Exception:
    gdf["geometry"] = gdf.buffer(0)

# --------------------------------------------------
# 4. Create unique ID
# --------------------------------------------------
gdf = gdf.reset_index(drop=True)

gdf["UNIQUE_ID"] = (
    gdf["COUNTYFP"].astype(str).str.strip() + "_" +
    gdf["VTDST"].astype(str).str.strip() + "_" +
    gdf["NAME"].astype(str).str.strip() + "_" +
    gdf.index.astype(str)
)

print("Rows in gdf:", len(gdf))
print("Unique UNIQUE_IDs:", gdf["UNIQUE_ID"].nunique())

# --------------------------------------------------
# 5. Create vote columns
# --------------------------------------------------
gdf["dem_votes"] = gdf["G20PREDBID"]
gdf["rep_votes"] = gdf["G20PRERTRU"]

gdf["other_votes"] = (
    gdf["G20PRELJOR"] +
    gdf["G20PREABLA"] +
    gdf["G20PREONON"]
)

gdf["two_party_votes"] = gdf["dem_votes"] + gdf["rep_votes"]
gdf["total_votes_all"] = gdf["dem_votes"] + gdf["rep_votes"] + gdf["other_votes"]

# --------------------------------------------------
# 6. Estimate population
# --------------------------------------------------
turnout_rate = 0.67
vap_share = 0.70

gdf["population"] = (
    gdf["total_votes_all"] / (turnout_rate * vap_share)
).fillna(0).round().astype(int)

# --------------------------------------------------
# 7. Keep only needed columns
# --------------------------------------------------
gdf_small = gdf[
    [
        "UNIQUE_ID",
        "STATEFP",
        "COUNTYFP",
        "COUNTY",
        "VTDST",
        "NAME",
        "dem_votes",
        "rep_votes",
        "other_votes",
        "two_party_votes",
        "total_votes_all",
        "population",
        "geometry",
    ]
].copy()

print("Nevada units:", len(gdf_small))

# --------------------------------------------------
# 8. Candidate adjacency pairs using intersects
# --------------------------------------------------
adj_df = gdf_small[["UNIQUE_ID", "geometry"]].copy()

cand = gpd.sjoin(
    adj_df,
    adj_df,
    how="inner",
    predicate="intersects"
)

cand = cand[cand["UNIQUE_ID_left"] != cand["UNIQUE_ID_right"]].copy()

cand["pair"] = cand.apply(
    lambda r: tuple(sorted((r["UNIQUE_ID_left"], r["UNIQUE_ID_right"]))),
    axis=1
)
cand = cand.drop_duplicates(subset="pair")

print("Candidate intersecting pairs:", len(cand))

# --------------------------------------------------
# 9. Keep only pairs with positive shared boundary length
# --------------------------------------------------
geom_map = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["geometry"]))

edge_rows = []
for _, row in cand.iterrows():
    u = row["UNIQUE_ID_left"]
    v = row["UNIQUE_ID_right"]

    shared = geom_map[u].boundary.intersection(geom_map[v].boundary)

    if shared.length > 0:
        edge_rows.append((u, v))

edges = pd.DataFrame(edge_rows, columns=["u", "v"])

print("Adjacency pairs with positive shared boundary:", len(edges))

# --------------------------------------------------
# 10. Build graph
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

print("\nInitial graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

isolated = list(nx.isolates(G))
print("Initial isolated nodes:", len(isolated))
print("First few isolated nodes:", isolated[:20])

components = list(nx.connected_components(G))
sizes = sorted([len(c) for c in components], reverse=True)

print("Initial connected components:", len(components))
print("Largest component size:", max(len(c) for c in components))
print("Largest 20 component sizes:", sizes[:20])

# --------------------------------------------------
# 11. Keep only the largest connected component
# --------------------------------------------------
largest_cc = max(nx.connected_components(G), key=len)

dropped_nodes = set(G.nodes()) - set(largest_cc)
print("\nDropping nodes outside largest component:", len(dropped_nodes))
print("First few dropped nodes:", list(dropped_nodes)[:20])

gdf_small = gdf_small[gdf_small["UNIQUE_ID"].isin(largest_cc)].copy()
edges = edges[
    edges["u"].isin(largest_cc) & edges["v"].isin(largest_cc)
].copy()

# --------------------------------------------------
# 12. Rebuild graph on largest component only
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

print("\nAfter keeping largest connected component:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Connected components:", nx.number_connected_components(G))
print("Is connected?", nx.is_connected(G))

# --------------------------------------------------
# 13. Optimization-ready objects
# --------------------------------------------------
neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

precincts = list(gdf_small["UNIQUE_ID"])
edges_list = list(zip(edges["u"], edges["v"]))
arcs = edges_list + [(v, u) for u, v in edges_list]

pop = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["population"]))
rep = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["rep_votes"]))
dem = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["dem_votes"]))

print("\nReady for optimization:")
print("Number of precincts:", len(precincts))
print("Number of undirected edges:", len(edges_list))
print("Number of directed arcs:", len(arcs))
print("Example precinct ID:", precincts[0] if precincts else None)

# Nevada congressional districts
p = 4

total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / p
tol = 0.10
L = (1 - tol) * target_pop
U = (1 + tol) * target_pop

print("\nPopulation targets for p = 4:")
print("Total population:", total_pop)
print("Target population:", target_pop)
print("Population lower bound:", L)
print("Population upper bound:", U)

total_rep_votes = sum(rep[i] for i in precincts)
total_dem_votes = sum(dem[i] for i in precincts)
total_votes = total_rep_votes + total_dem_votes

print("Total Republican votes:", total_rep_votes)
print("Total Democratic votes:", total_dem_votes)
print("Total two-party votes:", total_votes)

# --------------------------------------------------
# 14. Precompute incoming and outgoing arcs
# --------------------------------------------------
in_arcs = {i: [] for i in precincts}
out_arcs = {i: [] for i in precincts}

for (u, v) in arcs:
    out_arcs[u].append((u, v))
    in_arcs[v].append((u, v))

# --------------------------------------------------
# 15. Build Hess + SHIR contiguity model
# --------------------------------------------------
m = gp.Model("nevada_hess_contiguity_efficiency_gap")

# x[i,j] = 1 if precinct i is assigned to center j
x = m.addVars(precincts, precincts, vtype=GRB.BINARY, name="x")

# y[j] = 1 if precinct j is selected as a district center
y = m.addVars(precincts, vtype=GRB.BINARY, name="y")

# f[u,v,j] = flow of type j on directed arc (u,v)
f = m.addVars(arcs, precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="f")

# rep_win[j] = 1 if Republicans win district j
rep_win = m.addVars(precincts, vtype=GRB.BINARY, name="rep_win")

# u[j] = rep_win[j] * Tj
u_var = m.addVars(precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

# --------------------------------------------------
# 16. Hess constraints
# --------------------------------------------------
m.addConstrs(
    (gp.quicksum(x[i, j] for j in precincts) == 1 for i in precincts),
    name="assign"
)

m.addConstr(
    gp.quicksum(y[j] for j in precincts) == p,
    name="num_centers"
)

m.addConstrs(
    (x[i, j] <= y[j] for i in precincts for j in precincts),
    name="open_link"
)

m.addConstrs(
    (x[j, j] == y[j] for j in precincts),
    name="self_assign"
)

# --------------------------------------------------
# 17. Population constraints
# --------------------------------------------------
m.addConstrs(
    (gp.quicksum(pop[i] * x[i, j] for i in precincts) >= L * y[j]
     for j in precincts),
    name="pop_lb"
)

m.addConstrs(
    (gp.quicksum(pop[i] * x[i, j] for i in precincts) <= U * y[j]
     for j in precincts),
    name="pop_ub"
)

# --------------------------------------------------
# 18. SHIR flow contiguity constraints
# --------------------------------------------------
n = len(precincts)

for j in precincts:
    for i in precincts:
        if i == j:
            continue
        m.addConstr(
            gp.quicksum(f[u, v, j] for (u, v) in in_arcs[i]) -
            gp.quicksum(f[u, v, j] for (u, v) in out_arcs[i])
            == x[i, j],
            name=f"flow_balance_{i}_{j}"
        )

for j in precincts:
    for i in precincts:
        if i == j:
            continue
        m.addConstr(
            gp.quicksum(f[u, v, j] for (u, v) in in_arcs[i])
            <= (n - 1) * x[i, j],
            name=f"flow_in_cap_{i}_{j}"
        )

for j in precincts:
    m.addConstr(
        gp.quicksum(f[u, v, j] for (u, v) in in_arcs[j]) == 0,
        name=f"no_inflow_center_{j}"
    )

for j in precincts:
    for (u, v) in arcs:
        m.addConstr(
            f[u, v, j] <= (n - 1) * y[j],
            name=f"flow_open_{u}_{v}_{j}"
        )

for j in precincts:
    m.addConstr(
        gp.quicksum(f[u, v, j] for (u, v) in out_arcs[j]) ==
        gp.quicksum(x[i, j] for i in precincts if i != j),
        name=f"source_balance_{j}"
    )

# --------------------------------------------------
# 19. District-level vote totals
# --------------------------------------------------
Rj = {
    j: gp.quicksum(rep[i] * x[i, j] for i in precincts)
    for j in precincts
}

Dj = {
    j: gp.quicksum(dem[i] * x[i, j] for i in precincts)
    for j in precincts
}

Tj = {
    j: Rj[j] + Dj[j]
    for j in precincts
}

# --------------------------------------------------
# 20. Winner constraints
# --------------------------------------------------
Mvote = total_votes

for j in precincts:
    m.addConstr(rep_win[j] <= y[j], name=f"win_open_{j}")

    m.addConstr(
        Rj[j] - Dj[j] >= 1 - Mvote * (1 - rep_win[j]),
        name=f"rep_win_lb_{j}"
    )

    m.addConstr(
        Dj[j] - Rj[j] >= 1 - Mvote * rep_win[j],
        name=f"dem_win_lb_{j}"
    )

# --------------------------------------------------
# 21. Linearize u[j] = rep_win[j] * Tj[j]
# --------------------------------------------------
Vmax = total_votes

for j in precincts:
    m.addConstr(u_var[j] <= Vmax * rep_win[j], name=f"u1_{j}")
    m.addConstr(u_var[j] <= Tj[j], name=f"u2_{j}")
    m.addConstr(
        u_var[j] >= Tj[j] - Vmax * (1 - rep_win[j]),
        name=f"u3_{j}"
    )

# --------------------------------------------------
# 22. Wasted-vote expressions and efficiency gap objective
# --------------------------------------------------
WR = {
    j: Rj[j] - 0.5 * u_var[j]
    for j in precincts
}

WD = {
    j: Dj[j] - 0.5 * (Tj[j] - u_var[j])
    for j in precincts
}

m.setObjective(
    gp.quicksum(WD[j] - WR[j] for j in precincts),
    GRB.MAXIMIZE
)

# --------------------------------------------------
# 23. Solver settings
# --------------------------------------------------
m.Params.OutputFlag = 1
m.Params.TimeLimit = 200
m.Params.MIPFocus = 1

# --------------------------------------------------
# 24. Solve
# --------------------------------------------------
m.optimize()

# --------------------------------------------------
# 25. Extract solution
# --------------------------------------------------
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    if m.SolCount > 0:
        print("\nDistricting plan found.\n")

        centers = [j for j in precincts if y[j].X > 0.5]
        print(f"Number of centers selected: {len(centers)}")
        print("Centers:", centers)

        assignment = {}
        for i in precincts:
            assigned_center = None
            for j in precincts:
                if x[i, j].X > 0.5:
                    assigned_center = j
                    break
            assignment[i] = assigned_center

        center_to_district = {center: idx for idx, center in enumerate(centers)}
        district_assignment = {
            i: center_to_district[assignment[i]]
            for i in precincts
        }

        gdf_small["district"] = gdf_small["UNIQUE_ID"].map(district_assignment)

        print("\nDistrict statistics:")
        total_rep_wasted = 0.0
        total_dem_wasted = 0.0

        for center in centers:
            dnum = center_to_district[center]
            members = [i for i in precincts if assignment[i] == center]

            district_pop = sum(pop[i] for i in members)
            district_rep = sum(rep[i] for i in members)
            district_dem = sum(dem[i] for i in members)

            winner = "R" if rep_win[center].X > 0.5 else "D"

            district_WR = WR[center].getValue()
            district_WD = WD[center].getValue()

            total_rep_wasted += district_WR
            total_dem_wasted += district_WD

            print(
                f"District {dnum} (center {center}): "
                f"Pop={district_pop}, "
                f"R={district_rep}, D={district_dem}, "
                f"Winner={winner}, "
                f"WR={district_WR:.1f}, WD={district_WD:.1f}"
            )

        missing = gdf_small["district"].isnull().sum()
        print(f"\nUnassigned precincts: {missing}")

        eg_numerator = total_dem_wasted - total_rep_wasted
        eg = eg_numerator / total_votes if total_votes > 0 else None

        print("\n----- Efficiency Gap Results -----")
        print(f"Total Republican wasted votes: {total_rep_wasted:.2f}")
        print(f"Total Democratic wasted votes: {total_dem_wasted:.2f}")
        print(f"Efficiency gap numerator (D wasted - R wasted): {eg_numerator:.2f}")
        print(f"Efficiency gap: {eg:.6f}" if eg is not None else "Efficiency gap undefined")

    else:
        print("Solver stopped, but no incumbent solution was found.")
else:
    print(f"No feasible solution found. Solver status code: {m.status}")

