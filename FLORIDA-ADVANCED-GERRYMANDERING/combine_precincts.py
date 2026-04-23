import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict

# ==========================================
# 1. SETUP & TARGETS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

gdf = pd.read_pickle(os.path.join(OUTPUT_DIR, "fl_precincts_processed.pkl"))
with open(os.path.join(OUTPUT_DIR, "fl_graph.pkl"), "rb") as f:
    G = pickle.load(f)

# Project to equal-area CRS for accurate centroid distances (if not already)
if gdf.crs is None or gdf.crs.is_geographic:
    print("Reprojecting to EPSG:5070 for accurate distances...")
    gdf = gdf.to_crs(epsg=5070)

# --- Define Targets ---
# For p=28 Florida districts, ~420 super-precincts gives n/p ≈ 15 (sweet spot)
TARGET_SUPER_PRECINCTS = 500
TOTAL_POPULATION = gdf['population'].sum()
IDEAL_POP = TOTAL_POPULATION / TARGET_SUPER_PRECINCTS

POP_TOLERANCE_MAX = IDEAL_POP * 1.15  # Max 15% over ideal
POP_TOLERANCE_MIN = IDEAL_POP * 0.85  # Min 15% under ideal

# ==========================================
# 2. PREP DATA FOR FAST SEARCHING
# ==========================================
print(f"Targeting {TARGET_SUPER_PRECINCTS} groups. "
      f"Ideal population per group: {IDEAL_POP:,.0f}")

gdf['rep_pct'] = (gdf['rep_votes'] / gdf['two_party_votes']).fillna(0)
gdf['centroid'] = gdf.geometry.centroid

data_dict = gdf.set_index('UNIQUE_ID').to_dict('index')
unassigned = set(gdf['UNIQUE_ID'])
groups = {}
group_counter = 0

# ==========================================
# 3. SMART GROUPING LOGIC
# ==========================================
print("Growing super-precincts...")
while unassigned:
    # Pick an edge precinct (fewest unassigned neighbors) to avoid trapping
    seed = min(unassigned,
               key=lambda x: len([n for n in G.neighbors(x) if n in unassigned]))

    current_group = [seed]
    current_pop = data_dict[seed]['population']
    seed_centroid = data_dict[seed]['centroid']
    seed_rep_pct = data_dict[seed]['rep_pct']

    group_id = f"Group_{group_counter}"
    unassigned.remove(seed)

    while current_pop < POP_TOLERANCE_MAX:
        # Find all unassigned neighbors touching the current group
        potential_neighbors = set()
        for member in current_group:
            for nbr in G.neighbors(member):
                if nbr in unassigned:
                    potential_neighbors.add(nbr)

        if not potential_neighbors:
            break

        best_neighbor = None
        best_score = float('inf')

        for nbr in potential_neighbors:
            nbr_data = data_dict[nbr]
            if current_pop + nbr_data['population'] > POP_TOLERANCE_MAX:
                continue
            dist = seed_centroid.distance(nbr_data['centroid'])
            pol_diff = abs(seed_rep_pct - nbr_data['rep_pct'])
            score = dist * (1 + (pol_diff * 2))
            if score < best_score:
                best_score = score
                best_neighbor = nbr

        if best_neighbor:
            current_group.append(best_neighbor)
            current_pop += data_dict[best_neighbor]['population']
            unassigned.remove(best_neighbor)
        else:
            break

    groups[group_id] = current_group
    group_counter += 1

print(f"Initial groups formed: {len(groups)}")

# ==========================================
# 4. ORPHAN CLEANUP (fixed iterative version)
# ==========================================
print("Cleaning up undersized groups...")

# Start with every member mapped to its original group
final_group_mapping = {}
for gid, members in groups.items():
    for member in members:
        final_group_mapping[member] = gid

# Iteratively merge undersized groups into best-fit neighbors
changed = True
iterations = 0
MAX_ITERATIONS = 20

while changed and iterations < MAX_ITERATIONS:
    changed = False
    iterations += 1

    current_groups = defaultdict(list)
    for member, gid in final_group_mapping.items():
        current_groups[gid].append(member)

    group_pops = {
        gid: sum(data_dict[m]['population'] for m in members)
        for gid, members in current_groups.items()
    }

    for gid, members in list(current_groups.items()):
        if group_pops[gid] >= POP_TOLERANCE_MIN:
            continue

        # Count shared borders with each neighboring group
        neighbor_borders = defaultdict(int)
        for member in members:
            for nbr in G.neighbors(member):
                nbr_gid = final_group_mapping.get(nbr)
                if nbr_gid and nbr_gid != gid:
                    neighbor_borders[nbr_gid] += 1

        if not neighbor_borders:
            continue  # truly isolated

        # Prefer neighbors with room to grow
        def score_neighbor(ng):
            if group_pops.get(ng, 0) + group_pops[gid] > POP_TOLERANCE_MAX * 1.25:
                return -1
            return neighbor_borders[ng]

        best = max(neighbor_borders, key=score_neighbor)
        if score_neighbor(best) < 0:
            # All neighbors would overflow; merge into smallest anyway
            best = min(neighbor_borders, key=lambda ng: group_pops.get(ng, 0))

        for member in members:
            final_group_mapping[member] = best
        group_pops[best] = group_pops.get(best, 0) + group_pops[gid]
        group_pops[gid] = 0
        changed = True

print(f"Cleanup done after {iterations} iterations.")

# ==========================================
# 5. VERIFY CONNECTIVITY
# ==========================================
print("Verifying super-precinct connectivity...")
sp_members = defaultdict(list)
for member, gid in final_group_mapping.items():
    sp_members[gid].append(member)

disconnected = []
for gid, members in sp_members.items():
    if len(members) > 1:
        H = G.subgraph(members)
        if not nx.is_connected(H):
            disconnected.append((gid, len(members)))

if disconnected:
    print(f"⚠️  {len(disconnected)} super-precincts are disconnected. "
          f"Splitting them into connected components...")
    # Split disconnected super-precincts into their components
    next_id = max(int(gid.split("_")[1]) for gid in sp_members.keys()) + 1
    for gid, _ in disconnected:
        members = sp_members[gid]
        H = G.subgraph(members)
        comps = list(nx.connected_components(H))
        # Keep first component with original id, rename others
        for i, comp in enumerate(comps[1:], start=1):
            new_gid = f"Group_{next_id}"
            next_id += 1
            for member in comp:
                final_group_mapping[member] = new_gid
    print(f"Split complete. Now have {len(set(final_group_mapping.values()))} super-precincts.")
else:
    print("✅ All super-precincts are contiguous.")

gdf['super_precinct_id'] = gdf['UNIQUE_ID'].map(final_group_mapping)

# ==========================================
# 6. DISSOLVE, SAVE, & VISUALIZE
# ==========================================
print("Dissolving geometries...")
super_gdf = gdf.dissolve(
    by='super_precinct_id',
    aggfunc={'population': 'sum', 'rep_votes': 'sum', 'dem_votes': 'sum'}
).reset_index()

super_gdf['rep_pct'] = (
    super_gdf['rep_votes'] /
    (super_gdf['rep_votes'] + super_gdf['dem_votes']).replace(0, 1)
)

print(f"\nFinal count: {len(super_gdf)} super-precincts")
print(f"Largest pop: {super_gdf['population'].max():,.0f}")
print(f"Smallest pop: {super_gdf['population'].min():,.0f}")
print(f"Mean pop: {super_gdf['population'].mean():,.0f}")

# Final connectivity check on dissolved super-precincts
print("\nBuilding super-precinct adjacency for diagnostics...")
sp_adj = gpd.sjoin(
    super_gdf[['super_precinct_id', 'geometry']],
    super_gdf[['super_precinct_id', 'geometry']],
    how="inner", predicate="intersects"
)
sp_adj = sp_adj[sp_adj['super_precinct_id_left']
                != sp_adj['super_precinct_id_right']]

sp_G = nx.Graph()
for uid in super_gdf['super_precinct_id']:
    sp_G.add_node(uid)
for _, row in sp_adj.iterrows():
    sp_G.add_edge(row['super_precinct_id_left'],
                  row['super_precinct_id_right'])

if nx.is_connected(sp_G):
    print(f"✅ Super-precinct graph connected. "
          f"Diameter: {nx.diameter(sp_G)}, "
          f"Avg degree: {2*sp_G.number_of_edges()/sp_G.number_of_nodes():.1f}")
else:
    ccs = list(nx.connected_components(sp_G))
    print(f"⚠️  Super-precinct graph has {len(ccs)} components. "
          f"Largest: {max(len(c) for c in ccs)}")

# Save outputs
super_gdf.to_pickle(os.path.join(OUTPUT_DIR, "fl_super_precincts.pkl"))
print("Saved to fl_super_precincts.pkl")

super_gdf.to_file(os.path.join(OUTPUT_DIR, "fl_super_precincts.geojson"),
                  driver="GeoJSON")
print("Saved to fl_super_precincts.geojson")

fig, ax = plt.subplots(figsize=(10, 10))
super_gdf.plot(column='rep_pct', cmap='RdBu_r', edgecolor='white',
               linewidth=0.5, ax=ax)
ax.set_title(f"FL Super-Precincts (n={len(super_gdf)}) — "
             f"Colored by Republican vote share")
ax.set_axis_off()
plt.show()