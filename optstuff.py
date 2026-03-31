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


# --------------------------------------------------
# 2. Rebuild graph from edge list
# --------------------------------------------------
G = nx.Graph()

for uid in gdf_small["UNIQUE_ID"]:
    G.add_node(uid)

for _, row in edges.iterrows():
    G.add_edge(row["u"], row["v"])




precincts = list(gdf_small["UNIQUE_ID"])
edges_list = list(zip(edges["u"], edges["v"]))
arcs = edges_list + [(v, u) for u, v in edges_list]

K = range(27)   # example

pop = dict(zip(gdf_small["UNIQUE_ID"], gdf_small["population"]))

total_pop = sum(pop[i] for i in precincts)
target_pop = total_pop / len(K)
tol = 0.1
L = (1 - tol) * target_pop
U = (1 + tol) * target_pop

print(L,U)
