# Florida Congressional Redistricting Optimization

An optimization pipeline for exploring alternative Florida congressional district maps using 2024 precinct-level election data and 2020 census population counts. The pipeline ingests precinct geometries, area-weights census population onto them, aggregates the ~5,800 precincts into ~150 "super-precincts" to keep the MIP tractable, and then solves either a direct Gurobi MIP or a column-generation formulation to produce a districting plan that optimizes a chosen political objective (seat count or efficiency gap).

This is coursework for Decision Analytics Design. It is not intended for real-world map-drawing or advocacy.

---

## Requirements

You will need Python 3.10+ and a working Gurobi license. The pipeline leans on the following packages, all installable via pip: `geopandas`, `networkx`, `pandas`, `shapely`, `matplotlib`, `gurobipy`. On most systems:

```bash
pip install geopandas networkx pandas shapely matplotlib gurobipy
```

Gurobi is required for the two optimization scripts (`fl_advanced_gerrymander.py` and `fl_column_generation_copy.py`). A free academic WLS license is available from [gurobi.com/academia](https://www.gurobi.com/academia/) — sign up, and Gurobi will issue you a `WLSACCESSID`, `WLSSECRET`, and `LICENSEID`. You will paste these into the two optimization scripts (see "Configuring Gurobi" below).

---

## Data Sources and Download Instructions

The pipeline needs two shapefiles, both of which live in a `data/` folder at the project root. Neither is redistributed with this repo — you download them yourself.

### 1. Census Block Population (2020)

Used to apportion population counts onto precinct boundaries.

**Source:** US Census Bureau, TIGER/Line Shapefiles, 2020 tabulation blocks.

**Download URL:** [https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Blocks+%282020%29](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Blocks+%282020%29)

**Steps:**
1. Open the URL above.
2. In the "Select a State" dropdown, choose **Florida**.
3. Click **Submit**, then download the resulting zip. The file will be named `tl_2020_12_tabblock20.zip` (12 is Florida's FIPS code).
4. Unzip it and move the five core files (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) into the project's `data/` folder. The two `.xml` metadata files are optional and can be discarded.

This file is public-domain and has no license restrictions. Total population sums to exactly 21,538,187, matching the official 2020 Florida census count.

### 2. Florida 2024 Precinct-Level Election Results

Used for precinct boundaries and presidential vote counts.

**Source:** Voting and Election Science Team (VEST), 2025, *2024 Precinct-Level Election Results – Florida, V1.0.*

**Download:**
-  UF Election Lab:: [https://election.lab.ufl.edu/dataset/fl-2024-precinct-level-election-results/](https://election.lab.ufl.edu/dataset/fl-2024-precinct-level-election-results/)


Unzip and place the core shapefile bundle in `data/`. The expected filename is `fl_2024.shp` (plus its sibling `.shx`, `.dbf`, `.prj`, `.cpg` files). If your download uses a different filename, update the path near the top of `build_pop_precinct_data.py`.

---

## Expected Directory Layout

After downloading both datasets, your folder should look like this:

```
project_root/
├── data/
│   ├── fl_2024.shp                  (+ .shx, .dbf, .prj, .cpg)
│   └── tl_2020_12_tabblock20.shp    (+ .shx, .dbf, .prj, .cpg)
├── outputs/                         (created automatically)
├── build_pop_precinct_data.py
├── combine_precincts.py
├── population_to_precinct_check.py
├── fl_advanced_gerrymander.py
├── fl_column_generation_copy.py
└── README.md
```

The `outputs/` folder is created automatically on the first run and holds intermediate pickles and the final map artifacts.

---

## Configuring Gurobi

Both `fl_advanced_gerrymander.py` and `fl_column_generation_copy.py` contain a block near the top of the optimization section that looks like this:

```python
options = {
    "WLSACCESSID": "YOUR_ACCESS_ID_HERE",
    "WLSSECRET":   "YOUR_SECRET_HERE",
    "LICENSEID":   YOUR_LICENSE_ID_HERE,
}
env = gp.Env(params=options)
```

Replace the three placeholder values with the credentials from your Gurobi WLS license. If you already have a standard local Gurobi install and don't use WLS, you can delete the `options` dict and the `env=env` argument, leaving just `m = gp.Model("...")`.

---

## Running the Pipeline

Run the scripts from the project root, **in this order**. Each one depends on the outputs of the previous step.

### Step 1 — Build precinct dataset and adjacency graph

```bash
python build_pop_precinct_data.py
```

Loads the raw precinct and census shapefiles, computes area-weighted population per precinct via polygon intersection, builds a precinct adjacency graph, stitches any disconnected islands to the mainland by centroid distance, and writes `fl_precincts_processed.pkl`, `fl_graph.pkl`, and related files into `outputs/`. This step is the slowest — the census overlay against ~390k block polygons typically takes several minutes.

### Step 2 — Aggregate precincts into super-precincts

```bash
python combine_precincts.py
```

Greedily merges the ~5,800 precincts into ~150 population-balanced super-precincts, biasing merges toward compactness and political similarity. Writes `fl_super_precincts.pkl` and a matching GeoJSON. This is the key step that makes the downstream MIP tractable — without it, the optimization model is too large to solve.

### Step 3 (optional) — Sanity-check the data

```bash
python population_to_precinct_check.py
```

Reports total population, flags any super-precincts where recorded votes exceed population, and plots a population heatmap plus a votes-vs-population scatter. Run this if you want to verify the aggregation before committing compute time to the optimization step. The apportioned population total should land essentially on 21,538,187 (any gap is rounding from the integer cast).

### Step 4 — Run the optimization

You have two options here, and you should pick one based on the problem size you want to solve.

**Option A — Direct MIP (`fl_advanced_gerrymander.py`)** solves a compact indexed-district formulation with flow-based contiguity constraints. It works well at ~5 districts but scales poorly beyond that. Good for exploring small instances and understanding the formulation.

```bash
python fl_advanced_gerrymander.py
```

**Option B — Column generation (`fl_column_generation_copy.py`)** generates a large pool of candidate districts via randomized BFS partitions with partisan biases, then solves a set-partitioning master IP over that pool. This scales to Florida's full 28 congressional districts and is the recommended path for any serious run.

```bash
python fl_column_generation_copy.py
```

Both scripts will print a district-by-district summary table (population, vote shares, margins, winner, contiguity check), compute the statewide efficiency gap, and produce a matplotlib map of the final plan. The map and underlying data are saved to `outputs/fl_final_map.pkl` or `outputs/fl_final_map_colgen_v4.pkl` respectively.

---

## Tuning Knobs

A few parameters you may want to adjust depending on runtime vs. quality tradeoffs:

In `combine_precincts.py`, `TARGET_SUPER_PRECINCTS` controls the aggregation granularity. Lower values (e.g. 100) make the optimization faster but cruder; higher values (e.g. 200) preserve more geographic fidelity at the cost of a larger model.

In `fl_advanced_gerrymander.py`, `p` sets the number of districts and `tol` sets the population balance tolerance. The script is currently set to `p=5` with 17.5% tolerance to stay feasible; for a realistic 28-district plan with ~1% tolerance, use the column-generation script instead.

In `fl_column_generation_copy.py`, `N_PARTITIONS`, `PART_TIME_LIMIT`, `N_EXTRA`, and `EXTRA_TIME_LIMIT` control how many candidate partitions and individual districts are generated before the master IP runs. Increase for better solutions; decrease for faster turnaround. `OBJECTIVE_MODE` toggles between maximizing Republican seats (`"rep_wins"`) and maximizing the efficiency gap (`"eff_gap"`).

---

## Citation

If you reference this pipeline or its outputs in coursework, please cite the underlying data sources:

- U.S. Census Bureau. *2020 TIGER/Line Shapefiles: Tabulation Blocks.* Washington, DC: U.S. Department of Commerce.
- Voting and Election Science Team. 2025. *2024 Precinct-Level Election Results – Florida, V1.0.* UF Election Lab
