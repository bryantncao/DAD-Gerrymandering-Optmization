import geopandas as gpd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from shapely import wkb

# --------------------------------------------------
# 1. Read Florida precinct shapefile
# --------------------------------------------------
gdf = gpd.read_file(
    r"C:\Users\benne\Downloads\fl_2024_gen_prec\fl_2024_gen_prec\fl_2024_gen_all_prec\fl_2024_gen_all_prec.shp"
)

# --------------------------------------------------
# 2. Create vote columns
# --------------------------------------------------
gdf["dem_votes"] = gdf["G24PREDHAR"]
gdf["rep_votes"] = gdf["G24PRERTRU"]

# all presidential votes
gdf["other_votes"] = (
    gdf["G24PREASON"] +
    gdf["G24PRECTER"] +
    gdf["G24PREGSTE"] +
    gdf["G24PRELOLI"] +
    gdf["G24PREPCRU"]
)

gdf["two_party_votes"] = gdf["dem_votes"] + gdf["rep_votes"]
gdf["total_votes_all"] = gdf["dem_votes"] + gdf["rep_votes"] + gdf["other_votes"]

# --------------------------------------------------
# 3. Estimate total population and round to integers
# --------------------------------------------------
turnout_rate = 0.67
vap_share = 0.70

gdf["population"] = (
    gdf["total_votes_all"] / (turnout_rate * vap_share)
).fillna(0).round().astype(int)

# --------------------------------------------------
# 4. Keep only the columns we need
# --------------------------------------------------
gdf_small = gdf[
    [
        "UNIQUE_ID",
        "COUNTYFP",
        "CNTY_NAME",
        "PREC_ID",
        "dem_votes",
        "rep_votes",
        "other_votes",
        "two_party_votes",
        "total_votes_all",
        "population",
        "geometry",
    ]
].copy()



# --------------------------------------------------
# 6. Build adjacency pairs
# --------------------------------------------------
adj_df = gdf_small[["UNIQUE_ID", "geometry"]].copy()

adj = gpd.sjoin(
    adj_df,
    adj_df,
    how="inner",
    predicate="intersects"
)

# remove self-matches
adj = adj[adj["UNIQUE_ID_left"] != adj["UNIQUE_ID_right"]]

# --------------------------------------------------
# 7. Build statewide graph
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in adj.iterrows():
    G.add_edge(row["UNIQUE_ID_left"], row["UNIQUE_ID_right"])

print("Initial graph:")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

isolated = list(nx.isolates(G))
print("Initial isolated nodes:", len(isolated))
print("First few isolated nodes:", isolated[:20])

components = list(nx.connected_components(G))
print("Initial connected components:", len(components))
print("Largest component size:", max(len(c) for c in components))

# --------------------------------------------------
# 8. Build centroids in projected CRS for distance checks
# --------------------------------------------------
gdf_proj = gdf_small.to_crs(epsg=3086)  # Florida Albers
gdf_proj["centroid"] = gdf_proj.geometry.centroid

centroid_map = dict(zip(gdf_proj["UNIQUE_ID"], gdf_proj["centroid"]))

# --------------------------------------------------
# 9. Connect every small component to the main component
# --------------------------------------------------
components = list(nx.connected_components(G))
main_component = max(components, key=len)
main_nodes = list(main_component)

for comp in components:
    if comp == main_component:
        continue

    best_u = None
    best_v = None
    best_dist = float("inf")

    for u in comp:
        cu = centroid_map[u]
        for v in main_nodes:
            cv = centroid_map[v]
            dx = cu.x - cv.x
            dy = cu.y - cv.y
            dist = dx * dx + dy * dy

            if dist < best_dist:
                best_dist = dist
                best_u = u
                best_v = v

    G.add_edge(best_u, best_v)
    print(f"Connected component node {best_u} -> main component node {best_v}")

# --------------------------------------------------
# 10. Recompute components after connecting everything
# --------------------------------------------------
components = list(nx.connected_components(G))
print("\nAfter connecting components:")
print("Connected components:", len(components))
print("Largest component size:", max(len(c) for c in components))

isolated = list(nx.isolates(G))
print("Remaining isolated nodes:", len(isolated))

# --------------------------------------------------
# 11. Map component IDs back to precincts
# --------------------------------------------------
component_map = {}
for comp_id, comp in enumerate(components):
    for uid in comp:
        component_map[uid] = comp_id

gdf_small["component"] = gdf_small["UNIQUE_ID"].map(component_map)

comp_sizes = gdf_small["component"].value_counts().sort_values(ascending=False)
print("\nComponent sizes:")
print(comp_sizes.head(20))

# --------------------------------------------------
# 12. Save processed precinct dataset
# --------------------------------------------------
gdf_small.to_pickle(r"C:\Users\benne\Downloads\fl_precincts_processed.pkl")

gdf_small.to_file(
    r"C:\Users\benne\Downloads\fl_precincts_processed.geojson",
    driver="GeoJSON"
)

# --------------------------------------------------
# 13. Save graph edge list
# --------------------------------------------------
edge_list = pd.DataFrame(G.edges(), columns=["u", "v"])
edge_list.to_csv(
    r"C:\Users\benne\Downloads\fl_graph_edges.csv",
    index=False
)

# --------------------------------------------------
# 14. Save graph object and neighbors dictionary
# --------------------------------------------------
with open(r"C:\Users\benne\Downloads\fl_graph.pkl", "wb") as f:
    pickle.dump(G, f)

neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
with open(r"C:\Users\benne\Downloads\fl_neighbors.pkl", "wb") as f:
    pickle.dump(neighbors, f)

print("\nSaved:")
print("- fl_precincts_processed.pkl")
print("- fl_precincts_processed.geojson")
print("- fl_graph_edges.csv")
print("- fl_graph.pkl")
print("- fl_neighbors.pkl")

# --------------------------------------------------
# 15. Optional quick plot of final component map
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
gdf_small.plot(
    column="component",
    categorical=True,
    legend=False,
    ax=ax
)
ax.set_title("Florida Precinct Graph Components (After Connecting)")
ax.set_axis_off()
plt.show()
