import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. Load saved processed data and graph
# ------------------------------------------
gdf_small = pd.read_pickle(r"C:\Users\benne\Downloads\fl_precincts_processed.pkl")
edges = pd.read_csv(r"C:\Users\benne\Downloads\fl_graph_edges.csv")

# Make sure it's a GeoDataFrame
if not isinstance(gdf_small, gpd.GeoDataFrame):
    gdf_small = gpd.GeoDataFrame(gdf_small, geometry="geometry")

# Rebuild graph
G = nx.Graph()
for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])

# ------------------------------------------
# 2. Reproject for cleaner centroid plotting
# ------------------------------------------
gdf_plot = gdf_small.to_crs(epsg=3086).copy()
gdf_plot["centroid"] = gdf_plot.geometry.centroid

# ------------------------------------------
# 3. Choose county to inspect
# ------------------------------------------
county_name = "Duval"   # change this to any county name you want

county_df = gdf_plot[gdf_plot["CNTY_NAME"] == county_name].copy()

if county_df.empty:
    raise ValueError(f"No precincts found for county '{county_name}'")

county_ids = set(county_df["UNIQUE_ID"])
G_county = G.subgraph(county_ids).copy()

print(f"County: {county_name}")
print("Precincts:", len(county_df))
print("Graph nodes in county:", G_county.number_of_nodes())
print("Graph edges in county:", G_county.number_of_edges())
print("Connected components in county:", nx.number_connected_components(G_county))
print("Isolated nodes in county:", len(list(nx.isolates(G_county))))

# ------------------------------------------
# 4. Build centroid positions
# ------------------------------------------
pos = {
    row["UNIQUE_ID"]: (row["centroid"].x, row["centroid"].y)
    for _, row in county_df.iterrows()
}

# ------------------------------------------
# 5. Plot polygons + centroids + neighbor edges
# ------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# precinct boundaries
county_df.plot(
    ax=ax,
    color="lightgray",
    edgecolor="black",
    linewidth=0.4
)

# centroid points
county_df.set_geometry("centroid").plot(
    ax=ax,
    color="red",
    markersize=8
)

# graph edges between centroids
nx.draw_networkx_edges(
    G_county,
    pos=pos,
    ax=ax,
    edge_color="blue",
    width=0.7,
    alpha=0.8
)

plt.title(f"Precinct Centroids and Neighbor Edges: {county_name} County")
plt.axis("off")
plt.show()

# ------------------------------------------
# 6. Optional: inspect one sample precinct and its neighbors
# ------------------------------------------
sample_uid = county_df.iloc[0]["UNIQUE_ID"]
nbrs = list(G.neighbors(sample_uid))

print("\nSample precinct:", sample_uid)
print("Neighbors:", nbrs)

subset_ids = [sample_uid] + nbrs
subset = county_df[county_df["UNIQUE_ID"].isin(subset_ids)].copy()

fig, ax = plt.subplots(figsize=(8, 8))

county_df.plot(ax=ax, color="white", edgecolor="lightgray", linewidth=0.2)
subset.plot(ax=ax, color="yellow", edgecolor="black")
subset.set_geometry("centroid").plot(ax=ax, color="red", markersize=25)

# label just the sample precinct
sample_row = subset[subset["UNIQUE_ID"] == sample_uid].iloc[0]
ax.text(
    sample_row["centroid"].x,
    sample_row["centroid"].y,
    str(sample_row["PREC_ID"]),
    fontsize=9
)

plt.title(f"Sample Precinct and Its Neighbors: {county_name}")
plt.axis("off")
plt.show()
