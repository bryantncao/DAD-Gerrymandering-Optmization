
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

neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

precincts = list(gdf_small["UNIQUE_ID"])
edges_list = list(zip(edges["u"], edges["v"]))
arcs = edges_list + [(v, u) for u, v in edges_list]

pop = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["population"]))
rep = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["rep_votes"]))
dem = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["dem_votes"]))

print("\nReady for optimization on aggregated Nevada:")
print("Aggregated units:", len(precincts))
print("Undirected edges:", len(edges_list))
print("Directed arcs:", len(arcs))

print("\nEstimated model size after aggregation:")
print("x vars ~", len(precincts) * len(precincts))
print("f vars ~", len(arcs) * len(precincts))

# --------------------------------------------------
# 13. Optimization-ready objects
# --------------------------------------------------

# Nevada congressional districts
p = 4

total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / p
tol = 0.20
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
# 15. Precompute incoming and outgoing arcs
# --------------------------------------------------
in_arcs = {i: [] for i in precincts}
out_arcs = {i: [] for i in precincts}

for (u, v) in arcs:
    out_arcs[u].append((u, v))
    in_arcs[v].append((u, v))

# --------------------------------------------------
# 16. Build Hess + SHIR contiguity model
# --------------------------------------------------
'''
print("\nStarting model build...", flush=True)
m = gp.Model("nevada_hess_contiguity_efficiency_gap")

print("Adding x variables...", flush=True)
x = m.addVars(precincts, precincts, vtype=GRB.BINARY, name="x")
print("Finished x variables", flush=True)

print("Adding y variables...", flush=True)
y = m.addVars(precincts, vtype=GRB.BINARY, name="y")
print("Finished y variables", flush=True)

print("Adding f variables...", flush=True)
f = m.addVars(arcs, precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
print("Finished f variables", flush=True)

print("Adding rep_win variables...", flush=True)
rep_win = m.addVars(precincts, vtype=GRB.BINARY, name="rep_win")
print("Finished rep_win variables", flush=True)

print("Adding u variables...", flush=True)
u_var = m.addVars(precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
print("Finished u variables", flush=True)

# --------------------------------------------------
# 17. Hess constraints
# --------------------------------------------------
print("Adding assignment constraints...", flush=True)
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
print("Finished Hess constraints", flush=True)

# --------------------------------------------------
# 18. Population constraints
# --------------------------------------------------
print("Adding population constraints...", flush=True)
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
print("Finished population constraints", flush=True)

# --------------------------------------------------
# 19. SHIR flow contiguity constraints
# --------------------------------------------------
print("Adding flow constraints...", flush=True)
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
print("Finished flow constraints", flush=True)

# --------------------------------------------------
# 20. District-level vote totals
# --------------------------------------------------
print("Building district vote expressions...", flush=True)
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
print("Finished district vote expressions", flush=True)

# --------------------------------------------------
# 21. Winner constraints
# --------------------------------------------------
print("Adding winner constraints...", flush=True)
Mvote = total_votes

for j in precincts:
    m.addConstr(rep_win[j] <= y[j], name=f"win_open_{j}")

    # If rep_win[j] = 1, Republicans must win, but only if district is open
    m.addConstr(
        Rj[j] - Dj[j] >= 1 - Mvote * (1 - rep_win[j]),
        name=f"rep_win_lb_{j}"
    )

    # If rep_win[j] = 0 and district is open, Democrats must win
    m.addConstr(
        Dj[j] - Rj[j] >= 1 - Mvote * rep_win[j] - Mvote * (1 - y[j]),
        name=f"dem_win_lb_{j}"
    )
print("Finished winner constraints", flush=True)

# --------------------------------------------------
# 22. Linearize u[j] = rep_win[j] * Tj[j]
# --------------------------------------------------
print("Adding u linearization constraints...", flush=True)
Vmax = total_votes

for j in precincts:
    m.addConstr(u_var[j] <= Vmax * rep_win[j], name=f"u1_{j}")
    m.addConstr(u_var[j] <= Tj[j], name=f"u2_{j}")
    m.addConstr(
        u_var[j] >= Tj[j] - Vmax * (1 - rep_win[j]),
        name=f"u3_{j}"
    )
print("Finished u linearization constraints", flush=True)

# --------------------------------------------------
# 23. Wasted-vote expressions and efficiency gap objective
# --------------------------------------------------
print("Building objective...", flush=True)

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

m.setObjective(
    gp.quicksum(rep_win[j] for j in precincts),
    GRB.MAXIMIZE
)


print("Finished objective", flush=True)

# --------------------------------------------------
# 24. Solver settings
# --------------------------------------------------
m.Params.OutputFlag = OUTPUT_FLAG
m.Params.TimeLimit = TIME_LIMIT
m.Params.MIPFocus = MIP_FOCUS

# --------------------------------------------------
# 25. Solve
# --------------------------------------------------
print("\nAbout to call optimize()", flush=True)
m.optimize()

# --------------------------------------------------
# 26. Extract solution
# --------------------------------------------------
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    if m.SolCount > 0:
        print("\nDistricting plan found.\n", flush=True)

        centers = [j for j in precincts if y[j].X > 0.5]
        print(f"Number of centers selected: {len(centers)}", flush=True)
        print("Centers:", centers, flush=True)

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

        print("\nDistrict statistics:", flush=True)
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

            H = G.subgraph(members).copy()
            ncc = nx.number_connected_components(H)

            total_rep_wasted += district_WR
            total_dem_wasted += district_WD

            print(
                f"District {dnum} (center {center}): "
                f"Pop={district_pop}, "
                f"R={district_rep}, D={district_dem}, "
                f"Winner={winner}, "
                f"WR={district_WR:.1f}, WD={district_WD:.1f}, "
                f"GraphComponents={ncc}",
                flush=True
            )

        missing = gdf_small["district"].isnull().sum()
        print(f"\nUnassigned precincts: {missing}", flush=True)

        rep_seats = sum(1 for center in centers if rep_win[center].X > 0.5)
        print(f"Republican-won districts: {rep_seats} out of {len(centers)}", flush=True)

        eg_numerator = total_dem_wasted - total_rep_wasted
        eg = eg_numerator / total_votes if total_votes > 0 else None

        print("\n----- Reported Efficiency Gap Statistics -----", flush=True)
        print(f"Total Republican wasted votes: {total_rep_wasted:.2f}", flush=True)
        print(f"Total Democratic wasted votes: {total_dem_wasted:.2f}", flush=True)
        print(f"Efficiency gap numerator (D wasted - R wasted): {eg_numerator:.2f}", flush=True)
        print(f"Efficiency gap: {eg:.6f}" if eg is not None else "Efficiency gap undefined", flush=True)

        # --------------------------------------------------
        # Packing / cracking analysis
        # --------------------------------------------------
        district_summary = []

        for center in centers:
            dnum = center_to_district[center]
            members = [i for i in precincts if assignment[i] == center]

            district_pop = sum(pop[i] for i in members)
            district_rep = sum(rep[i] for i in members)
            district_dem = sum(dem[i] for i in members)
            district_total = district_rep + district_dem

            rep_share = district_rep / district_total if district_total > 0 else 0.0
            dem_share = district_dem / district_total if district_total > 0 else 0.0
            margin = rep_share - dem_share

            winner = "R" if rep_win[center].X > 0.5 else "D"

            district_WR = WR[center].getValue()
            district_WD = WD[center].getValue()

            district_summary.append({
                "district": dnum,
                "center": center,
                "population": district_pop,
                "rep_votes": district_rep,
                "dem_votes": district_dem,
                "total_votes": district_total,
                "rep_share": rep_share,
                "dem_share": dem_share,
                "margin": margin,
                "winner": winner,
                "rep_wasted": district_WR,
                "dem_wasted": district_WD,
            })

        district_df = pd.DataFrame(district_summary).sort_values("district").reset_index(drop=True)

        print("\n----- District Summary Table -----", flush=True)
        print(
            district_df[
                [
                    "district", "winner", "population",
                    "rep_votes", "dem_votes",
                    "rep_share", "dem_share", "margin",
                    "rep_wasted", "dem_wasted"
                ]
            ].to_string(index=False),
            flush=True
        )

        # --------------------------------------------------
        # Statewide summary
        # --------------------------------------------------
        statewide_rep_share = total_rep_votes / total_votes if total_votes > 0 else 0.0
        statewide_dem_share = total_dem_votes / total_votes if total_votes > 0 else 0.0

        rep_seats = int((district_df["winner"] == "R").sum())
        dem_seats = int((district_df["winner"] == "D").sum())

        rep_seat_share = rep_seats / len(district_df) if len(district_df) > 0 else 0.0
        dem_seat_share = dem_seats / len(district_df) if len(district_df) > 0 else 0.0

        print("\n----- Statewide Summary -----", flush=True)
        print(f"Republican statewide vote share: {statewide_rep_share:.3f}", flush=True)
        print(f"Democratic statewide vote share: {statewide_dem_share:.3f}", flush=True)
        print(f"Republican seat share: {rep_seat_share:.3f} ({rep_seats} / {len(district_df)})", flush=True)
        print(f"Democratic seat share: {dem_seat_share:.3f} ({dem_seats} / {len(district_df)})", flush=True)

        # --------------------------------------------------
        # Heuristic packing / cracking flags
        # --------------------------------------------------
        # "Packed" district heuristic:
        # winner gets >= 65% of the two-party vote
        packed_thresh = 0.65

        # "Cracked" district heuristic:
        # losing party still has substantial support, say 40% to 49%
        crack_low = 0.40
        crack_high = 0.49

        packed_R = district_df[(district_df["winner"] == "R") & (district_df["rep_share"] >= packed_thresh)].copy()
        packed_D = district_df[(district_df["winner"] == "D") & (district_df["dem_share"] >= packed_thresh)].copy()

        cracked_R = district_df[
            (district_df["winner"] == "D") &
            (district_df["rep_share"] >= crack_low) &
            (district_df["rep_share"] <= crack_high)
        ].copy()

        cracked_D = district_df[
            (district_df["winner"] == "R") &
            (district_df["dem_share"] >= crack_low) &
            (district_df["dem_share"] <= crack_high)
        ].copy()

        print("\n----- Heuristic Packing / Cracking Flags -----", flush=True)

        if len(packed_R) > 0:
            print("Possible packed Republican districts:", flush=True)
            print(packed_R[["district", "rep_share", "margin", "rep_wasted"]].to_string(index=False), flush=True)
        else:
            print("No obvious packed Republican districts under current threshold.", flush=True)

        if len(packed_D) > 0:
            print("Possible packed Democratic districts:", flush=True)
            print(packed_D[["district", "dem_share", "margin", "dem_wasted"]].to_string(index=False), flush=True)
        else:
            print("No obvious packed Democratic districts under current threshold.", flush=True)

        if len(cracked_R) > 0:
            print("Possible cracked Republican districts (R loses but still has substantial vote share):", flush=True)
            print(cracked_R[["district", "rep_share", "margin", "rep_wasted"]].to_string(index=False), flush=True)
        else:
            print("No obvious cracked Republican districts under current threshold.", flush=True)

        if len(cracked_D) > 0:
            print("Possible cracked Democratic districts (D loses but still has substantial vote share):", flush=True)
            print(cracked_D[["district", "dem_share", "margin", "dem_wasted"]].to_string(index=False), flush=True)
        else:
            print("No obvious cracked Democratic districts under current threshold.", flush=True)

        # --------------------------------------------------
        # Simple textual interpretation
        # --------------------------------------------------
        print("\n----- Basic Interpretation -----", flush=True)

        if rep_seat_share > statewide_rep_share + 0.10:
            print("Republicans appear to outperform their statewide vote share in seat share.", flush=True)
        elif dem_seat_share > statewide_dem_share + 0.10:
            print("Democrats appear to outperform their statewide vote share in seat share.", flush=True)
        else:
            print("Seat share is not dramatically different from statewide vote share.", flush=True)

        if len(packed_D) > 0 and len(cracked_D) > 0:
            print("Pattern suggests Democrats may be both packed into some districts and cracked across others.", flush=True)
        elif len(packed_R) > 0 and len(cracked_R) > 0:
            print("Pattern suggests Republicans may be both packed into some districts and cracked across others.", flush=True)
        elif len(packed_D) > 0:
            print("Pattern suggests some Democratic packing.", flush=True)
        elif len(packed_R) > 0:
            print("Pattern suggests some Republican packing.", flush=True)
        elif len(cracked_D) > 0:
            print("Pattern suggests some Democratic cracking.", flush=True)
        elif len(cracked_R) > 0:
            print("Pattern suggests some Republican cracking.", flush=True)
        else:
            print("No strong packing/cracking signal under these simple thresholds.", flush=True)

        # --------------------------------------------------
        # Plot districts
        # --------------------------------------------------
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        gdf_small.plot(
            column="district",
            categorical=True,
            cmap="tab20",
            linewidth=0.25,
            edgecolor="black",
            legend=True,
            ax=ax
        )

        ax.set_title("Optimized Districts")
        ax.set_axis_off()
        plt.show()

    else:
        print("Solver stopped, but no incumbent solution was found.", flush=True)
else:
    print(f"No feasible solution found. Solver status code: {m.status}", flush=True)
'''
# --------------------------------------------------
# Indexed-district formulation
# --------------------------------------------------
print("\nStarting indexed-district model build...", flush=True)

districts = list(range(p))
n = len(precincts)

m = gp.Model("indexed_district_contiguity")

# --------------------------------------------------
# 1. Variables
# --------------------------------------------------
print("Adding x variables...", flush=True)
x = m.addVars(precincts, districts, vtype=GRB.BINARY, name="x")
print("Finished x variables", flush=True)

print("Adding r variables...", flush=True)
r = m.addVars(precincts, districts, vtype=GRB.BINARY, name="r")
print("Finished r variables", flush=True)

print("Adding f variables...", flush=True)
f = m.addVars(arcs, districts, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
print("Finished f variables", flush=True)

print("Adding district size variables...", flush=True)
s = m.addVars(districts, lb=0.0, ub=n, vtype=GRB.CONTINUOUS, name="s")
print("Finished district size variables", flush=True)

print("Adding root-supply helper variables...", flush=True)
g = m.addVars(precincts, districts, lb=0.0, ub=n, vtype=GRB.CONTINUOUS, name="g")
print("Finished root-supply helper variables", flush=True)

print("Adding rep_win variables...", flush=True)
rep_win = m.addVars(districts, vtype=GRB.BINARY, name="rep_win")
print("Finished rep_win variables", flush=True)

print("Adding u variables...", flush=True)
u_var = m.addVars(districts, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
print("Finished u variables", flush=True)

# --------------------------------------------------
# 2. Assignment constraints
# --------------------------------------------------
print("Adding assignment constraints...", flush=True)

m.addConstrs(
    (gp.quicksum(x[i, k] for k in districts) == 1 for i in precincts),
    name="assign"
)

m.addConstrs(
    (gp.quicksum(r[i, k] for i in precincts) == 1 for k in districts),
    name="one_root"
)

m.addConstrs(
    (r[i, k] <= x[i, k] for i in precincts for k in districts),
    name="root_in_district"
)

print("Finished assignment/root constraints", flush=True)

print("Adding root-order symmetry breaking...", flush=True)

prec_order = {i: idx for idx, i in enumerate(precincts)}

root_index = {
    k: gp.quicksum(prec_order[i] * r[i, k] for i in precincts)
    for k in districts
}

for k in range(p - 1):
    m.addConstr(
        root_index[k] <= root_index[k + 1],
        name=f"root_order_{k}"
    )

print("Finished root-order symmetry breaking", flush=True)

print("Adding district size definition constraints...", flush=True)
m.addConstrs(
    (s[k] == gp.quicksum(x[i, k] for i in precincts) for k in districts),
    name="district_size"
)
print("Finished district size definition constraints", flush=True)

# --------------------------------------------------
# 3. Population balance
# --------------------------------------------------
print("Adding population constraints...", flush=True)

m.addConstrs(
    (gp.quicksum(pop[i] * x[i, k] for i in precincts) >= L for k in districts),
    name="pop_lb"
)

m.addConstrs(
    (gp.quicksum(pop[i] * x[i, k] for i in precincts) <= U for k in districts),
    name="pop_ub"
)

print("Finished population constraints", flush=True)

# --------------------------------------------------
# 4. Contiguity via one flow network per district
# --------------------------------------------------
print("Adding flow constraints...", flush=True)

M = n

for k in districts:
    for i in precincts:
        m.addConstr(g[i, k] <= s[k], name=f"g_ub_size_{i}_{k}")
        m.addConstr(g[i, k] <= M * r[i, k], name=f"g_ub_root_{i}_{k}")
        m.addConstr(g[i, k] >= s[k] - M * (1 - r[i, k]), name=f"g_lb_{i}_{k}")
        m.addConstr(g[i, k] >= 0, name=f"g_nonneg_{i}_{k}")

for k in districts:
    for i in precincts:
        m.addConstr(
            gp.quicksum(f[u, v, k] for (u, v) in in_arcs[i]) -
            gp.quicksum(f[u, v, k] for (u, v) in out_arcs[i])
            == x[i, k] - g[i, k],
            name=f"flow_balance_{i}_{k}"
        )

for k in districts:
    for i in precincts:
        m.addConstr(
            gp.quicksum(f[u, v, k] for (u, v) in in_arcs[i])
            <= (n - 1) * x[i, k],
            name=f"flow_in_cap_{i}_{k}"
        )

for k in districts:
    for (u, v) in arcs:
        m.addConstr(
            f[u, v, k] <= (n - 1) * x[u, k],
            name=f"flow_from_assign_{u}_{v}_{k}"
        )
        m.addConstr(
            f[u, v, k] <= (n - 1) * x[v, k],
            name=f"flow_to_assign_{u}_{v}_{k}"
        )

print("Finished flow constraints", flush=True)

# --------------------------------------------------
# 5. District-level vote totals
# --------------------------------------------------
print("Building district vote expressions...", flush=True)

Rk = {
    k: gp.quicksum(rep[i] * x[i, k] for i in precincts)
    for k in districts
}

Dk = {
    k: gp.quicksum(dem[i] * x[i, k] for i in precincts)
    for k in districts
}

Tk = {
    k: Rk[k] + Dk[k]
    for k in districts
}

print("Finished district vote expressions", flush=True)

# --------------------------------------------------
# 6. Winner constraints
# --------------------------------------------------
print("Adding winner constraints...", flush=True)

total_rep_votes = sum(rep[i] for i in precincts)
total_dem_votes = sum(dem[i] for i in precincts)
total_votes = total_rep_votes + total_dem_votes
Mvote = total_votes if total_votes > 0 else 1

# Republicans win district k if rep_win[k] = 1
# Ties are allowed here; switch 0 to 1 if you want strict wins
for k in districts:
    m.addConstr(
        Rk[k] - Dk[k] >= 0 - Mvote * (1 - rep_win[k]),
        name=f"rep_win_lb_{k}"
    )
    m.addConstr(
        Dk[k] - Rk[k] >= 0 - Mvote * rep_win[k],
        name=f"dem_win_lb_{k}"
    )

print("Finished winner constraints", flush=True)

# --------------------------------------------------
# 7. Optional EG reporting machinery
# --------------------------------------------------
print("Adding u linearization constraints...", flush=True)

Vmax = total_votes if total_votes > 0 else 1

for k in districts:
    m.addConstr(u_var[k] <= Vmax * rep_win[k], name=f"u1_{k}")
    m.addConstr(u_var[k] <= Tk[k], name=f"u2_{k}")
    m.addConstr(
        u_var[k] >= Tk[k] - Vmax * (1 - rep_win[k]),
        name=f"u3_{k}"
    )

print("Finished u linearization constraints", flush=True)

WR = {
    k: Rk[k] - 0.5 * u_var[k]
    for k in districts
}

WD = {
    k: Dk[k] - 0.5 * (Tk[k] - u_var[k])
    for k in districts
}

# --------------------------------------------------
# 8. Objective toggle
# --------------------------------------------------
OBJECTIVE_MODE = "rep_wins"   # "rep_wins" or "eff_gap"

print("Building objective...", flush=True)
if OBJECTIVE_MODE == "rep_wins":
    m.setObjective(
        gp.quicksum(rep_win[k] for k in districts),
        GRB.MAXIMIZE
    )
elif OBJECTIVE_MODE == "eff_gap":
    m.setObjective(
        gp.quicksum(WD[k] - WR[k] for k in districts),
        GRB.MAXIMIZE
    )
else:
    raise ValueError("Unknown OBJECTIVE_MODE")
print("Finished objective", flush=True)

# --------------------------------------------------
# 9. Solver settings
# --------------------------------------------------
m.Params.OutputFlag = 1
m.Params.TimeLimit = 1000
m.Params.MIPFocus = 1


# --------------------------------------------------
# 10. Solve
# --------------------------------------------------
print("\nAbout to call optimize()", flush=True)
m.optimize()

# --------------------------------------------------
# 11. Extract solution
# --------------------------------------------------
if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
    if m.SolCount > 0:
        print("\nDistricting plan found.\n", flush=True)

        assignment = {}
        for i in precincts:
            assigned_k = None
            for k in districts:
                if x[i, k].X > 0.5:
                    assigned_k = k
                    break
            assignment[i] = assigned_k

        gdf_small["district"] = gdf_small["UNIQUE_ID"].map(assignment)

        print("\nDistrict statistics:", flush=True)
        total_rep_wasted = 0.0
        total_dem_wasted = 0.0

        for k in districts:
            members = [i for i in precincts if assignment[i] == k]

            district_pop = sum(pop[i] for i in members)
            district_rep = sum(rep[i] for i in members)
            district_dem = sum(dem[i] for i in members)

            winner = "R" if rep_win[k].X > 0.5 else "D"

            district_WR = WR[k].getValue()
            district_WD = WD[k].getValue()

            H = G.subgraph(members).copy()
            ncc = nx.number_connected_components(H) if len(members) > 0 else 0

            total_rep_wasted += district_WR
            total_dem_wasted += district_WD

            district_total = district_rep + district_dem
            rep_share = district_rep / district_total if district_total > 0 else 0.0
            dem_share = district_dem / district_total if district_total > 0 else 0.0
            margin = rep_share - dem_share

            print(
                f"District {k}: "
                f"Pop={district_pop}, "
                f"R={district_rep}, D={district_dem}, "
                f"RepShare={rep_share:.3f}, DemShare={dem_share:.3f}, "
                f"Margin={margin:.3f}, "
                f"Winner={winner}, "
                f"WR={district_WR:.1f}, WD={district_WD:.1f}, "
                f"GraphComponents={ncc}",
                flush=True
            )

        missing = gdf_small["district"].isnull().sum()
        print(f"\nUnassigned precincts: {missing}", flush=True)

        rep_seats = sum(1 for k in districts if rep_win[k].X > 0.5)
        print(f"Republican-won districts: {rep_seats} out of {len(districts)}", flush=True)

        eg_numerator = total_dem_wasted - total_rep_wasted
        eg = eg_numerator / total_votes if total_votes > 0 else None

        print("\n----- Reported Efficiency Gap Statistics -----", flush=True)
        print(f"Total Republican wasted votes: {total_rep_wasted:.2f}", flush=True)
        print(f"Total Democratic wasted votes: {total_dem_wasted:.2f}", flush=True)
        print(f"Efficiency gap numerator (D wasted - R wasted): {eg_numerator:.2f}", flush=True)
        print(f"Efficiency gap: {eg:.6f}" if eg is not None else "Efficiency gap undefined", flush=True)

    else:
        print("Solver stopped, but no incumbent solution was found.", flush=True)
else:
    print(f"No feasible solution found. Solver status code: {m.status}", flush=True)

