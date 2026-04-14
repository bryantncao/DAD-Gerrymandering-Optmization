import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import gurobipy as gp
from gurobipy import GRB


# --------------------------------------------------
# 1. Load saved files
# --------------------------------------------------
gdf_small = pd.read_pickle(r"C:\Users\benne\Downloads\fl_precincts_processed.pkl")
edges = pd.read_csv(r"C:\Users\benne\Downloads\fl_graph_edges.csv")

with open(r"C:\Users\benne\Downloads\fl_graph.pkl", "rb") as f:
    G_saved = pickle.load(f)

with open(r"C:\Users\benne\Downloads\fl_neighbors.pkl", "rb") as f:
    neighbors_saved = pickle.load(f)


# --------------------------------------------------
# 2. Rebuild graph from edge list
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])


# --------------------------------------------------
# 3. Sets and parameters
# --------------------------------------------------
precincts = list(gdf_small["UNIQUE_ID"])
edges_list = list(zip(edges["u"], edges["v"]))
arcs = edges_list + [(v, u) for u, v in edges_list]

p = 28   # number of districts

# population
pop = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["population"]))

total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / p
tol = 0.10
L = (1 - tol) * target_pop
U = (1 + tol) * target_pop

print("Population lower bound:", L)
print("Population upper bound:", U)

# vote data
rep = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["rep_votes"]))
dem = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["dem_votes"]))

total_rep_votes = sum(rep[i] for i in precincts)
total_dem_votes = sum(dem[i] for i in precincts)
total_votes = total_rep_votes + total_dem_votes

print("Total Republican votes:", total_rep_votes)
print("Total Democratic votes:", total_dem_votes)
print("Total two-party votes:", total_votes)

# Precompute incoming and outgoing arcs
in_arcs = {i: [] for i in precincts}
out_arcs = {i: [] for i in precincts}

for (u, v) in arcs:
    out_arcs[u].append((u, v))
    in_arcs[v].append((u, v))


# --------------------------------------------------
# 4. Build Hess + SHIR contiguity model
# --------------------------------------------------
m = gp.Model("hess_contiguity_efficiency_gap")

# x[i,j] = 1 if precinct i is assigned to center j
x = m.addVars(precincts, precincts, vtype=GRB.BINARY, name="x")

# y[j] = 1 if precinct j is selected as a district center
y = m.addVars(precincts, vtype=GRB.BINARY, name="y")

# f[u,v,j] = flow of type j on directed arc (u,v)
f = m.addVars(arcs, precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="f")

# rep_win[j] = 1 if Republicans win district j
rep_win = m.addVars(precincts, vtype=GRB.BINARY, name="rep_win")

# u[j] = rep_win[j] * Tj, used to linearize wasted vote expressions
u_var = m.addVars(precincts, lb=0.0, vtype=GRB.CONTINUOUS, name="u")


# --------------------------------------------------
# 5. Hess constraints
# --------------------------------------------------

# Each precinct assigned to exactly one center
m.addConstrs(
    (gp.quicksum(x[i, j] for j in precincts) == 1 for i in precincts),
    name="assign"
)

# Exactly p centers
m.addConstr(
    gp.quicksum(y[j] for j in precincts) == p,
    name="num_centers"
)

# Can only assign to an open center
m.addConstrs(
    (x[i, j] <= y[j] for i in precincts for j in precincts),
    name="open_link"
)

# If j is chosen as a center, it assigns itself to itself
m.addConstrs(
    (x[j, j] == y[j] for j in precincts),
    name="self_assign"
)


# --------------------------------------------------
# 6. Population constraints
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
# 7. SHIR flow contiguity constraints
# --------------------------------------------------
n = len(precincts)

# Flow balance at non-center nodes:
# sum_in f[u,i,j] - sum_out f[i,v,j] = x[i,j]
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

# A precinct can only receive type-j flow if assigned to j
for j in precincts:
    for i in precincts:
        if i == j:
            continue
        m.addConstr(
            gp.quicksum(f[u, v, j] for (u, v) in in_arcs[i])
            <= (n - 1) * x[i, j],
            name=f"flow_in_cap_{i}_{j}"
        )

# No type-j flow enters its own center j
for j in precincts:
    m.addConstr(
        gp.quicksum(f[u, v, j] for (u, v) in in_arcs[j]) == 0,
        name=f"no_inflow_center_{j}"
    )

# Optional tightening: if j is not an open center, no type-j flow can exist
for j in precincts:
    for (u, v) in arcs:
        m.addConstr(
            f[u, v, j] <= (n - 1) * y[j],
            name=f"flow_open_{u}_{v}_{j}"
        )

# Stronger source constraint at the center:
# center j sends one unit of flow for every OTHER precinct assigned to j
for j in precincts:
    m.addConstr(
        gp.quicksum(f[u, v, j] for (u, v) in out_arcs[j]) ==
        gp.quicksum(x[i, j] for i in precincts if i != j),
        name=f"source_balance_{j}"
    )


# --------------------------------------------------
# 8. District-level vote totals
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
# 9. Winner constraints
# --------------------------------------------------
# Big-M for vote margin constraints
Mvote = total_votes

for j in precincts:
    # unopened districts cannot have a winner
    m.addConstr(rep_win[j] <= y[j], name=f"win_open_{j}")

    # if rep_win[j] = 1, Republicans must win district j
    m.addConstr(
        Rj[j] - Dj[j] >= 1 - Mvote * (1 - rep_win[j]),
        name=f"rep_win_lb_{j}"
    )

    # if rep_win[j] = 0, Democrats must win district j
    m.addConstr(
        Dj[j] - Rj[j] >= 1 - Mvote * rep_win[j],
        name=f"dem_win_lb_{j}"
    )


# --------------------------------------------------
# 10. Linearize u[j] = rep_win[j] * Tj[j]
# --------------------------------------------------
Vmax = total_votes  # safe but loose upper bound

for j in precincts:
    m.addConstr(u_var[j] <= Vmax * rep_win[j], name=f"u1_{j}")
    m.addConstr(u_var[j] <= Tj[j], name=f"u2_{j}")
    m.addConstr(
        u_var[j] >= Tj[j] - Vmax * (1 - rep_win[j]),
        name=f"u3_{j}"
    )


# --------------------------------------------------
# 11. Wasted-vote expressions and efficiency gap objective
# --------------------------------------------------
# If Republicans win:
#   WR = R - T/2
#   WD = D
# If Democrats win:
#   WR = R
#   WD = D - T/2
#
# Using u = rep_win * T:
#   WR = R - 0.5*u
#   WD = D - 0.5*(T - u)

WR = {
    j: Rj[j] - 0.5 * u_var[j]
    for j in precincts
}

WD = {
    j: Dj[j] - 0.5 * (Tj[j] - u_var[j])
    for j in precincts
}

# Republican-favoring efficiency gap objective:
# maximize Democratic wasted votes - Republican wasted votes
m.setObjective(
    gp.quicksum(WD[j] - WR[j] for j in precincts),
    GRB.MAXIMIZE
)


# --------------------------------------------------
# 12. Solver settings
# --------------------------------------------------
m.Params.OutputFlag = 1
m.Params.TimeLimit = 100
m.Params.MIPFocus = 1


# --------------------------------------------------
# 13. Solve
# --------------------------------------------------
m.optimize()


# --------------------------------------------------
# 14. Extract solution
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

        # Relabel chosen centers to district numbers 0,1,...,p-1 for plotting
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
            district_total_votes = district_rep + district_dem

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

        # --------------------------------------------------
        # 15. Plot districts
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 12))

        gdf_small.plot(
            column="district",
            cmap="tab20",
            linewidth=0.1,
            edgecolor="black",
            ax=ax,
            legend=True
        )

        plt.title("Hess Districting Plan with Contiguity and Efficiency Gap Objective")
        plt.axis("off")
        plt.show()

    else:
        print("Solver stopped, but no incumbent solution was found.")

else:
    print(f"No feasible solution found. Solver status code: {m.status}")