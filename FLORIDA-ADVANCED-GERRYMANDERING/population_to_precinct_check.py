import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import os

# ==========================================
# PATH SETUP & DATA LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

print("Loading processed SUPER-PRECINCT data...")
gdf = pd.read_pickle(os.path.join(OUTPUT_DIR, "fl_super_precincts.pkl"))

if not isinstance(gdf, gpd.GeoDataFrame):
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

gdf['total_votes_all'] = gdf['rep_votes'] + gdf['dem_votes']

# ==========================================
# 1. STATEWIDE TOTALS
# ==========================================
total_pop = gdf['population'].sum()
total_votes = gdf['total_votes_all'].sum()
overall_turnout = total_votes / total_pop if total_pop > 0 else 0

print("\n" + "="*50)
print("🌎 STATEWIDE REALITY CHECK")
print("="*50)
print(f"Apportioned Population: {total_pop:,.0f}")
print(f"Actual 2020 FL Census:  21,538,187")
print(f"Population Difference:  {abs(21538187 - total_pop):,.0f}")
if abs(21538187 - total_pop) < 5000:
    print("   -> ✅ PERFECT! No significant population lost.")
else:
    print("   -> ⚠️  Large population gap detected.")

print(f"\nTotal Presidential Votes: {total_votes:,.0f}")
print(f"Implied Turnout:          {overall_turnout:.1%}")

# ==========================================
# 2. SOLVER-READINESS CHECKS
# ==========================================
print("\n" + "="*50)
print("🔍 SOLVER-READINESS CHECKS")
print("="*50)

p = 28
target_pop = total_pop / p
print(f"For p={p} districts, target pop per district: {target_pop:,.0f}")
print(f"At 5% tolerance:  {0.95*target_pop:,.0f} to {1.05*target_pop:,.0f}")
print(f"At 10% tolerance: {0.90*target_pop:,.0f} to {1.10*target_pop:,.0f}")

# n/p ratio check
n_over_p = len(gdf) / p
print(f"\nn/p ratio: {n_over_p:.1f}")
if n_over_p < 8:
    print("   -> ⚠️  Low — may struggle with population balance")
elif n_over_p > 25:
    print("   -> ⚠️  High — solver will be slow")
else:
    print("   -> ✅ Good range for tractability")

# Largest super-precinct vs ideal district pop
max_sp = gdf['population'].max()
if max_sp > target_pop * 0.5:
    print(f"   -> ⚠️  Largest super-precinct ({max_sp:,.0f}) is > 50% of "
          f"target district pop. May cause balance issues.")

# ==========================================
# 3. ADJACENCY GRAPH DIAGNOSTICS
# ==========================================
print("\nBuilding adjacency graph for diagnostics...")
sp_adj = gpd.sjoin(
    gdf[['super_precinct_id', 'geometry']],
    gdf[['super_precinct_id', 'geometry']],
    how="inner", predicate="intersects"
)
sp_adj = sp_adj[sp_adj['super_precinct_id_left']
                != sp_adj['super_precinct_id_right']]

G = nx.Graph()
for uid in gdf['super_precinct_id']:
    G.add_node(uid)
for _, row in sp_adj.iterrows():
    G.add_edge(row['super_precinct_id_left'],
               row['super_precinct_id_right'])

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Connected: {nx.is_connected(G)}")
if nx.is_connected(G):
    diameter = nx.diameter(G)
    avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"Diameter: {diameter}")
    print(f"Avg degree: {avg_deg:.1f}")
    if diameter > 40:
        print("   -> ⚠️  High diameter — graph may be stringy, slower to solve")
    if avg_deg < 3:
        print("   -> ⚠️  Low avg degree — few merge options")
else:
    ccs = list(nx.connected_components(G))
    print(f"⚠️  {len(ccs)} components. Sizes: {sorted([len(c) for c in ccs], reverse=True)[:5]}")

# ==========================================
# 4. ANOMALIES
# ==========================================
print("\n" + "="*50)
print("🚨 ANOMALIES")
print("="*50)

print("\n📈 Top 5 Most Populated:")
top_5 = gdf[['super_precinct_id', 'population']].sort_values(
    'population', ascending=False).head(5)
print(top_5.to_string(index=False))

valid = gdf[gdf['population'] > 0].copy()
valid['vote_to_pop_ratio'] = valid['total_votes_all'] / valid['population']
anomalies = valid[valid['vote_to_pop_ratio'] > 1.0]

if len(anomalies) == 0:
    print("\n✅ No super-precincts have more votes than residents.")
else:
    print(f"\n⚠️  {len(anomalies)} anomalies (votes > population):")
    worst = anomalies.sort_values('vote_to_pop_ratio', ascending=False).head(5)
    print(worst[['super_precinct_id', 'population', 'total_votes_all']].to_string(index=False))

ghosts = gdf[(gdf['population'] == 0) & (gdf['total_votes_all'] > 0)]
if len(ghosts) > 0:
    print(f"\n👻 {len(ghosts)} ghost super-precincts (votes but no population)")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
print("\n" + "="*50)
print("📊 Generating Visualizations...")

fig1, ax1 = plt.subplots(figsize=(10, 10))
gdf.plot(column='population', cmap='OrRd', linewidth=0.5,
         edgecolor='gray', legend=True,
         legend_kwds={'label': "Population by Super-Precinct"}, ax=ax1)
ax1.set_title("Florida Super-Precincts: Population", fontsize=14, fontweight='bold')
ax1.set_axis_off()

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(valid['population'], valid['total_votes_all'],
            alpha=0.4, color='blue', edgecolor='black', s=30)
max_val = max(valid['population'].max(), valid['total_votes_all'].max())
ax2.plot([0, max_val], [0, max_val], 'r--', label='100% voted (impossible)')
ax2.plot([0, max_val], [0, max_val * 0.5], 'g-', label='50% voted (expected)')
ax2.set_title("Super-Precinct Population vs Total Votes",
              fontsize=14, fontweight='bold')
ax2.set_xlabel("Population")
ax2.set_ylabel("Total Presidential Votes")
ax2.legend()
plt.grid(True, alpha=0.3)

plt.show()