import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle

# --------------------------------------------------
# 1. Load saved files
# --------------------------------------------------
gdf_small = pd.read_pickle(r"C:\Users\benne\Downloads\fl_precincts_processed.pkl")
edges = pd.read_csv(r"C:\Users\benne\Downloads\fl_graph_edges.csv")

with open(r"C:\Users\benne\Downloads\fl_graph.pkl", "rb") as f:
    G_saved = pickle.load(f)

with open(r"C:\Users\benne\Downloads\fl_neighbors.pkl", "rb") as f:
    neighbors_saved = pickle.load(f)

print("Files loaded successfully.\n")

# --------------------------------------------------
# 2. Rebuild graph from edge list
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

# --------------------------------------------------
# 3. Basic dataframe checks
# --------------------------------------------------
print("=== DATAFRAME CHECKS ===")
print("Shape:", gdf_small.shape)
print("\nColumns:")
print(list(gdf_small.columns))

print("\nHead:")
print(gdf_small.head())

print("\nDtypes:")
print(gdf_small.dtypes)

# --------------------------------------------------
# 4. Sanity checks on key columns
# --------------------------------------------------
print("\n=== KEY COLUMN CHECKS ===")
required_cols = [
    "UNIQUE_ID",
    "dem_votes",
    "rep_votes",
    "other_votes",
    "two_party_votes",
    "total_votes_all",
    "population",
    "geometry",
]

missing_cols = [col for col in required_cols if col not in gdf_small.columns]
print("Missing required columns:", missing_cols)

print("\nMissing values by required column:")
print(gdf_small[required_cols].isna().sum())

print("\nDuplicate UNIQUE_IDs:", gdf_small["UNIQUE_ID"].duplicated().sum())

# --------------------------------------------------
# 5. Vote and population totals
# --------------------------------------------------
print("\n=== TOTALS ===")
print("Total estimated population:", gdf_small["population"].sum())
print("Total Dem votes:", gdf_small["dem_votes"].sum())
print("Total Rep votes:", gdf_small["rep_votes"].sum())
print("Total Other votes:", gdf_small["other_votes"].sum())
print("Total two-party votes:", gdf_small["two_party_votes"].sum())
print("Total all presidential votes:", gdf_small["total_votes_all"].sum())

# consistency checks
print("\n=== CONSISTENCY CHECKS ===")
two_party_diff = (
    gdf_small["two_party_votes"] - (gdf_small["dem_votes"] + gdf_small["rep_votes"])
).abs().sum()
print("Two-party vote mismatch total:", two_party_diff)

all_vote_diff = (
    gdf_small["total_votes_all"]
    - (gdf_small["dem_votes"] + gdf_small["rep_votes"] + gdf_small["other_votes"])
).abs().sum()
print("All-vote mismatch total:", all_vote_diff)

print("Any negative populations?", (gdf_small["population"] < 0).any())
print("Population dtype:", gdf_small["population"].dtype)

# --------------------------------------------------
# 6. Graph checks
# --------------------------------------------------
print("\n=== GRAPH CHECKS ===")
print("Rebuilt graph nodes:", G.number_of_nodes())
print("Rebuilt graph edges:", G.number_of_edges())

print("Saved graph nodes:", G_saved.number_of_nodes())
print("Saved graph edges:", G_saved.number_of_edges())

df_nodes = set(gdf_small["UNIQUE_ID"])
graph_nodes = set(G.nodes())
saved_graph_nodes = set(G_saved.nodes())

print("\nNode alignment checks:")
print("Missing in rebuilt graph:", len(df_nodes - graph_nodes))
print("Extra in rebuilt graph:", len(graph_nodes - df_nodes))
print("Missing in saved graph:", len(df_nodes - saved_graph_nodes))
print("Extra in saved graph:", len(saved_graph_nodes - df_nodes))

bad_edges = edges[
    (~edges["u"].isin(gdf_small["UNIQUE_ID"])) |
    (~edges["v"].isin(gdf_small["UNIQUE_ID"]))
]
print("\nBad edges in CSV:", len(bad_edges))

# --------------------------------------------------
# 7. Connectivity checks
# --------------------------------------------------
print("\n=== CONNECTIVITY CHECKS ===")
components = list(nx.connected_components(G))
print("Connected components (rebuilt graph):", len(components))
print("Largest component size (rebuilt graph):", max(len(c) for c in components))

isolated = list(nx.isolates(G))
print("Isolated nodes (rebuilt graph):", len(isolated))
print("First few isolated nodes:", isolated[:20])

components_saved = list(nx.connected_components(G_saved))
print("\nConnected components (saved graph):", len(components_saved))
print("Largest component size (saved graph):", max(len(c) for c in components_saved))

isolated_saved = list(nx.isolates(G_saved))
print("Isolated nodes (saved graph):", len(isolated_saved))
print("First few isolated nodes (saved graph):", isolated_saved[:20])

# --------------------------------------------------
# 8. Compare rebuilt graph to saved graph
# --------------------------------------------------
print("\n=== GRAPH COMPARISON ===")
rebuilt_edges = set(tuple(sorted(e)) for e in G.edges())
saved_edges = set(tuple(sorted(e)) for e in G_saved.edges())

print("Edges in rebuilt but not saved:", len(rebuilt_edges - saved_edges))
print("Edges in saved but not rebuilt:", len(saved_edges - rebuilt_edges))

# --------------------------------------------------
# 9. Neighbor dictionary checks
# --------------------------------------------------
print("\n=== NEIGHBOR DICTIONARY CHECKS ===")
print("Neighbor dict size:", len(neighbors_saved))

missing_neighbor_keys = df_nodes - set(neighbors_saved.keys())
extra_neighbor_keys = set(neighbors_saved.keys()) - df_nodes

print("Missing neighbor keys:", len(missing_neighbor_keys))
print("Extra neighbor keys:", len(extra_neighbor_keys))

sample_node = list(gdf_small["UNIQUE_ID"])[0]
print("\nSample node:", sample_node)
print("Neighbors from dict:", neighbors_saved.get(sample_node, []))
print("Neighbors from saved graph:", list(G_saved.neighbors(sample_node)))
print("Neighbors from rebuilt graph:", list(G.neighbors(sample_node)))

# --------------------------------------------------
# 10. Geometry / plotting check
# --------------------------------------------------
print("\n=== GEOMETRY CHECKS ===")
print("Geometry type:", type(gdf_small))
print("Has geometry column:", "geometry" in gdf_small.columns)

# convert to GeoDataFrame if needed
if not isinstance(gdf_small, gpd.GeoDataFrame):
    gdf_small = gpd.GeoDataFrame(gdf_small, geometry="geometry")

print("GeoDataFrame confirmed.")
print("CRS:", gdf_small.crs)



# --------------------------------------------------
# 12. Final summary
# --------------------------------------------------
print("\n=== FINAL SUMMARY ===")
all_good = True

if missing_cols:
    all_good = False
if gdf_small["UNIQUE_ID"].duplicated().sum() > 0:
    all_good = False
if len(df_nodes - graph_nodes) > 0 or len(graph_nodes - df_nodes) > 0:
    all_good = False
if len(bad_edges) > 0:
    all_good = False
if len(components) != 1:
    all_good = False
if len(isolated) != 0:
    all_good = False

if all_good:
    print("Everything looks good. Your saved data and graph are ready for optimization.")
else:
    print("Some checks failed. Review the output above before optimizing.")


# --------------------------------------------------
# 11. Quick plot
# --------------------------------------------------
print("\n=== PLOTTING CHECK ===")
fig, ax = plt.subplots(figsize=(10, 10))
gdf_small.plot(ax=ax, color="lightgray", edgecolor="black")
ax.set_title("Florida Precincts Loaded from Saved Files")
ax.set_axis_off()
plt.show()
