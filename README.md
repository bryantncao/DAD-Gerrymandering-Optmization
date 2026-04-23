# Redistricting Optimization — Nevada & Florida

Coursework project for Decision Analytics Design. The overall goal is to draw congressional districts by solving a mixed-integer program that takes real precinct geometries, population counts, and vote totals as input, and produces a districting plan that maximizes some political objective (seat count or efficiency gap) while enforcing population balance and contiguity.

The project covers two states. Nevada is the small proof-of-concept — 4 congressional districts and about 1,900 precincts, tractable as a direct MIP. Florida is the main focus, and comes in two versions. The scripts in this top-level folder are the **main Florida implementation**, using a direct Hess + SHIR MIP formulation on the raw precinct data. The `FLORIDA-ADVANCED-GERRYMANDERING/` subfolder is an **improved version** that scales better, using precinct aggregation and column generation, and also swaps in cleaner data sources. Both are kept in the repo so the two approaches can be compared; which one you should read depends on what you're trying to do.

---

## Folder Layout

```
project_root/
├── nevada_original_optimization.py
├── nevada_new_formulation.py
├── nevada_subset_new_formulation_shown_in_presentation.py
├── florida_dataframe_generator.py
├── original_optimization_feasibility_check.py
├── florida_optimization_too_big_didn_t_work.py
├── sanitychecker.py
├── plotter.py
├── resources/                         ← supplementary redistricting references
├── FLORIDA-ADVANCED-GERRYMANDERING/   ← improved Florida pipeline, own README
├── README_FLORIDA-REDISTRCTING-ADVANCED.md
└── README.md                          ← this file
```

---

## Requirements

Python 3.10+ and a Gurobi license. Install the Python dependencies with:

```bash
pip install geopandas networkx pandas shapely matplotlib gurobipy
```

A free academic WLS license is available from [gurobi.com/academia](https://www.gurobi.com/academia/). It will issue you a `WLSACCESSID`, `WLSSECRET`, and `LICENSEID`. Several of the optimization scripts instantiate a plain `gp.Model(...)` and rely on Gurobi picking up credentials from your environment; if yours are WLS-only, wrap the model construction in an explicit env block:

```python
options = {
    "WLSACCESSID": "YOUR_ACCESS_ID_HERE",
    "WLSSECRET":   "YOUR_SECRET_HERE",
    "LICENSEID":   YOUR_LICENSE_ID_HERE,
}
env = gp.Env(params=options)
m = gp.Model("model_name", env=env)
```

---

## Data Sources

The project pulls from different sources depending on which scripts you're running. Each of the three sections below corresponds to a distinct group of scripts.

### Nevada

All Nevada scripts read from a VEST 2020 Nevada precinct shapefile (`nv_2020.shp`). VEST's 2016–2020 datasets are free under CC BY-NC-ND 4.0 and available from UF's election lab: [https://election.lab.ufl.edu/data-archive/](https://election.lab.ufl.edu/data-archive/). Search for "Nevada 2020" and download the shapefile bundle.

### Florida (top-level scripts)

The top-level Florida scripts (`florida_dataframe_generator.py`, `florida_optimization_too_big_didn_t_work.py`, `original_optimization_feasibility_check.py`, and the two utilities) read from a VEST 2024 Florida shapefile (`fl_2024_gen_all_prec.shp`) redistributed through the Redistricting Data Hub: [https://redistrictingdatahub.org/](https://redistrictingdatahub.org/). Create a free account, search the Florida 2024 general election precinct results, and download the zip.

### Florida-ADVANCED-GERRYMANDERING

The scripts in the ADVANCED-GERRYMANDERING folder use the VEST 2024 Florida shapefile (`fl_2024_gen_all_prec.shp`), available through UF Election Lab: [https://election.lab.ufl.edu/precinct-data/](https://election.lab.ufl.edu/precinct-data/). The advanced subfolder uses a cleaner version of the same data — see the subfolder's README for details.

The advanced pipeline also requires 2020 census block population data from the US Census Bureau's TIGER/Line shapefiles. Instructions for downloading that are in the advanced folder's README.

### Hardcoded paths

Every script at this top level points to absolute Windows paths like `r"C:\Users\benne\Downloads\..."`. Before running anything, open the script and update the paths near the top to match wherever you downloaded the data locally. This is the single most common cause of run-time failure.

---

## Nevada Scripts

Nevada was the proof-of-concept that the mathematical formulation actually works. At 4 districts and ~1,900 precincts, the direct MIP is small enough that Gurobi finds feasible plans in minutes.

`nevada_original_optimization.py` implements the classical **Hess model** with **SHIR (Shirabe) flow-based contiguity**. In the Hess formulation, each district is identified by a "center" precinct, and the variable `x[i, j]` means "precinct i is assigned to the district whose center is precinct j." Contiguity is enforced by requiring that every assigned precinct can route one unit of flow back to its center through within-district edges. The script solves for an efficiency-gap-maximizing plan and reports district-by-district wasted vote counts.

`nevada_new_formulation.py` is the updated version using an **indexed-district formulation**. Instead of Hess-style `x[i, j]`, it uses `x[i, k]` where `k ∈ {0, 1, ..., p-1}` is a district index. Each district picks a single root precinct, and flow-based contiguity is adapted accordingly. This breaks the symmetry in the Hess model (where any precinct in a district could serve as its "center") via a root-ordering constraint, which makes the branch-and-bound tree smaller. The file contains both the Hess version (triple-quote commented out) and the new formulation so you can compare. Set `OBJECTIVE_MODE` at the bottom to either `"rep_wins"` (maximize Republican seats) or `"eff_gap"` (maximize the Republican-favoring efficiency gap).

`nevada_subset_new_formulation_shown_in_presentation.py` is a BFS-based subset of Nevada used for live demos. The config block at the top lets you set `SUBSET_N` (how many precincts to carve out, starting from `START_NODE`) and `SUBSET_P` (how many districts to draw over them). With `SUBSET_N=50, SUBSET_P=2` the model solves in a few seconds, which is what made it usable in the course presentation.

---

## Florida Scripts — Main Implementation

The Florida scripts at this level are the direct port of the Nevada formulation to Florida. They share the same Hess + SHIR + efficiency-gap structure, run on VEST 2024 data redistributed through RDH, and estimate population with a turnout-times-voting-age-share heuristic (as opposed to proper census interpolation, which is what the advanced folder does).

`florida_dataframe_generator.py` is the main Florida data builder. It loads the VEST 2024 shapefile, creates vote columns, and estimates population as `total_votes / (0.67 × 0.70)` — i.e., assumes ~67% turnout among ~70% of the voting-age population. It then builds the precinct adjacency graph, stitches disconnected islands to the mainland by centroid distance, and pickles the result to disk.

`original_optimization_feasibility_check.py` is a minimal feasibility probe. It strips the model down to just the assignment constraint and the population upper/lower bounds, sets the objective to zero, and asks Gurobi whether any plan exists that satisfies both. Useful as a quick check before adding expensive contiguity or political constraints.

`florida_optimization_too_big_didn_t_work.py` applies the full Hess + SHIR + efficiency-gap formulation to Florida's ~5,800 precincts with 28 districts. The filename is honest — the resulting model is too large for Gurobi to solve on typical hardware in a reasonable time window. It was the experience of running this and watching it fail that motivated the super-precinct aggregation approach in the advanced folder. The script is kept here both as documentation of the baseline and so anyone studying the scaling problem can reproduce the observation.

---

## Utilities

`sanitychecker.py` validates the pickles produced by `florida_dataframe_generator.py`. It checks that required columns exist, flags missing values and duplicate IDs, verifies that `two_party_votes` and `total_votes_all` are internally consistent, compares the rebuilt graph to the saved graph for edge and node alignment, counts connected components, and plots the full state to confirm the GeoDataFrame still renders. Run this immediately after `florida_dataframe_generator.py` to catch problems before wasting time on optimization.

`plotter.py` is a debugging visualization tool. Given a county name (default: Duval), it pulls that county's precincts out of the processed pickle, subsets the adjacency graph to just those nodes, and plots the polygons with red centroid dots and blue neighbor-edge lines overlaid. This is how you verify the adjacency graph actually captures geographic neighbors and isn't missing edges or inventing bogus ones. A second plot highlights one sample precinct and its immediate neighbors.

---

## The Advanced Pipeline

The `FLORIDA-ADVANCED-GERRYMANDERING/` subfolder contains a scaled-up version of the Florida analysis. It improves on the main implementation in three ways. It replaces the turnout-based population estimate with area-weighted interpolation from 2020 census block data, so population numbers reflect actual residents rather than a heuristic. It greedily aggregates the ~5,800 precincts into ~150 compactness- and politics-aware "super-precincts," which shrinks the MIP enough to be tractable. And it uses a column-generation formulation — building a large pool of candidate districts via randomized BFS with partisan biases, then solving a set-partitioning master IP over the pool — which scales to the full 28 districts where the direct MIP does not.

That folder has its own README (`README_FLORIDA-REDISTRCTING-ADVANCED.md`, kept at the top level for visibility) covering the data sources in detail, the execution order of its internal scripts, and the parameters you can tune. Read it if you want to produce a full 28-district Florida plan rather than reproduce the main-implementation limits.

---

## Running Order

The scripts here are independent explorations tied together by a common theme, not a single linear pipeline. Here's the sensible order for someone trying to understand the project from scratch.

First, work through the Nevada scripts to see the formulation working end-to-end. Run `nevada_subset_new_formulation_shown_in_presentation.py` with small subset settings to see a complete solve in seconds, then bump up to `nevada_new_formulation.py` for the full state.

Next, look at the Florida main implementation. Run `florida_dataframe_generator.py` to build the pickles, then `sanitychecker.py` to confirm they look correct, then `plotter.py` on any county to confirm adjacency is sensible. You can optionally kick off `original_optimization_feasibility_check.py` to confirm population balance is achievable. Running `florida_optimization_too_big_didn_t_work.py` is optional — it will not finish on typical hardware, but it's worth kicking off briefly to see the model-build log if you're studying the scaling problem.

Finally, move into `FLORIDA-ADVANCED-GERRYMANDERING/` and follow the instructions in that folder's README for the improved pipeline.

---

## Citations

If you reference this work, cite the underlying data sources:

- U.S. Census Bureau. *2020 TIGER/Line Shapefiles: Tabulation Blocks.* Washington, DC: U.S. Department of Commerce. (Used in the advanced subfolder only.)
- Voting and Election Science Team. 2025. *2024 Precinct-Level Election Results – Florida, V1.0.* UF Election Lab. (Advanced subfolder.)
- Voting and Election Science Team. 2025. *2024 Precinct-Level Election Results – Florida, V1.0.* Redistributed via Redistricting Data Hub. (Top-level Florida scripts.)
- Voting and Election Science Team. *2020 Precinct-Level Election Results – Nevada.* UF Election Lab Data Archive.
