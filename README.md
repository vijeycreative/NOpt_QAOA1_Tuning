
# Code, Benchmarks, and Numerical Data for Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models  
  
This repository contains the source code, benchmark datasets, and numerical results used in the research paper:  
  
> V Vijendran, Dax Enshan Koh, Eunok Bae, Hyukjoon Kwon, Ping Koy Lam, Syed M Assad  
> *Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models*  
> https://arxiv.org/abs/2501.16419  
  
The repository provides all materials needed to reproduce and analyze the numerical results presented in the paper, including:  
  
1. **Source code** for all algorithms benchmarked: QAOA, RQAOA, Iter-QAOA, and SDP.  
2. **Benchmark datasets** containing generated graph instances with and without external fields, together with the best solution found by Gurobi for a fixed walltime until the incumbent solution converged, along with the corresponding MIP gap.  
3. **Simulation results** stored in JSON format containing the outputs of the benchmarked algorithms. Due to the large file size, the simulation-results folder is distributed as a compressed `7z` archive.  
4. **Jupyter notebooks** used to generate the numerical plots reported in the paper.  
5. **Auxiliary utilities** for graph generation, exact classical solvers, recursive graph elimination, and polynomial root solving.  
  
This repository is intended to serve as a reference and benchmarking resource for researchers studying combinatorial optimization, Ising models, and variational quantum algorithms.  
  
  
## Repository Overview  
  
```text  
.  
├── benchmark_data/  
│ ├── er_*.gz  
│ ├── erf_*.gz  
│ └── ...  
│  
├── code/  
│ ├── fqs.py  
│ ├── graphs.py  
│ ├── RQAOA.py  
│ ├── RQAOA_Fields.py  
│ ├── solvers.py  
│ ├── utils.py  
│ └── Subdivision_Alg_Test.ipynb  
│  
├── plot_scripts/  
│ ├── Adversarial Graphs/  
│ │ ├── First_Local_Extremum_Analysis_...  
│ │ └── First_Local_Extremum_Analysis_...  
| │ └── ...  
│ ├── 2-Local-Opt_Path.ipynb  
│ ├── 2-Local-Conc.ipynb  
│ ├── Max_Freq_Diff.ipynb  
│ ├── RQAOA_128_Comp.ipynb  
│ ├── RQAOA_256_Comp.ipynb  
│ └── RQAOA_Fields.ipynb  
│  
├── simulation_data.7z  
├── LICENSE  
└── README.md
```


## File and Folder Description

## `code/`
This folder contains the main algorithmic implementations and supporting utilities used throughout the paper.

* **`graphs.py`**: Graph-generation and graph-I/O utilities. This file includes routines for generating benchmark graph families such as Erdős-Rényi, Sherrington-Kirkpatrick, bipartite, and regular graphs, as well as helpers for storing and loading weighted graph instances together with classical reference solutions.
* **`utils.py`**: Shared utilities used primarily by the recursive algorithms. This file contains graph-management routines for variable elimination, assignment reconstruction, graph visualization helpers, and functions for extracting reduced graph data into array form.
* **`RQAOA.py`**: Main implementation of Recursive QAOA (RQAOA) for weighted Ising models without external fields. It includes the closed-form level-1 QAOA expectation formulas, the reduced one-dimensional optimization over `gamma`, analytic recovery of `beta`, and the recursive elimination loop.
* **`RQAOA_Fields.py`**: Recursive QAOA for weighted Ising models **with external fields**. In addition to edge-based elimination, this implementation also allows direct node elimination using local field biases. Iter-QAOA can be activated from this file by setting `only_node=True` in the `RQAOA_Fields(...)` routine.
* **`solvers.py`**: Classical benchmark solvers and relaxations, including:
    * Exact **Max-Cut** via Gurobi.
    * Exact **2-local Ising ground-state solving** via Gurobi.
    * Exact **2-local Ising with external fields** via Gurobi.
    * **SDP-based** relaxation with random hyperplane rounding.
* **`fqs.py`**: Fast cubic and quartic polynomial root solvers used as a helper utility in the parameter-optimization pipeline.
    * *Important attribution*: This file was not developed by V. Vijendran. It is derived from the third-party repository: [NKrvasica, fqs: Fast Quartic and Cubic Solver](https://github.com/NKrvasica/fqs).
    * In this project, `fqs.py` is used as a supporting utility for solving quartic equations that arise in the analytic or semi-analytic reduction of the level-1 QAOA optimization problem.
* **`Subdivision_Alg_Test.ipynb`**: A verification notebook for the subdivision algorithm discussed in the paper.

## `benchmark_data/`
This folder contains the benchmark graph instances used in the numerical study. A total of **400 benchmark instances** are included across the problem classes studied in the paper.

### Naming Convention
The benchmark filenames follow a compact naming convention.

| Example Filename | Interpretation |
| :--- | :--- |
| `er_128_0.1_1.gz` | **Erdős-Rényi** graph with 128 vertices, 0.1 edge probability/density, and instance number 1. |
| `erf_128_0.1_1.gz` | **Erdős-Rényi graph with external fields**, with the same parameter interpretation. |

**General Rules:**
* `er`: Erdős-Rényi graph without external fields.
* `erf`: Erdős-Rényi graph with external fields.
* The final integer in the filename denotes the instance index.

> The benchmark files store the generated weighted graph instances together with the best solution found by Gurobi under a fixed walltime (once the incumbent solution had converged) as well as the associated MIP gap.

## `plot_scripts/`
This folder contains the Jupyter notebooks and scripts used to generate the plots reported in the paper. These notebooks reproduce the plots and comparisons reported in the paper from the stored benchmark and simulation data.

### Correspondence between scripts and paper figures:

| Paper Figure | Corresponding Script(s) |
| :--- | :--- |
| **Figure 2** | `2-Local-Opt_Path.ipynb` |
| **Figure 3** | `2-Local-Conc.ipynb` |
| **Figure 4** | Scripts inside `plot_scripts/Adversarial Graphs/` |
| **Figure 5** | `RQAOA_128_Comp.ipynb` and `RQAOA_256_Comp.ipynb` |
| **Figure 6** | `RQAOA_Fields.ipynb` |
| **Figure 7** | `Max_Freq_Diff.ipynb` |

## `simulation_data.7z`
This archive contains the simulation results produced by running the benchmarked algorithms on the graph instances in `benchmark_data/`.

* The extracted simulation files are stored in **JSON** format and follow the same naming convention as the benchmark instances (e.g., `er_128_0.1_1.json`, `erf_128_0.1_1.json`).
* This makes it straightforward to match each simulation output file to its corresponding input graph instance.
* The simulation archive is compressed because the full collection of numerical outputs is large.

## Notes on Reproducibility
This repository is intended to make the numerical study as reproducible as possible by providing:

* The benchmark instances,
* The main algorithm implementations,
* The classical reference solvers,
* The simulation outputs, and
* The plotting notebooks used to generate the figures.

The benchmark and simulation filenames are matched by construction, so each output file can be directly associated with its corresponding input instance.

Because the simulation results are already included, users can either:
1. Regenerate the figures directly from the stored outputs, or
2. Rerun the algorithms themselves on the benchmark instances.

## Third-Party Code Attribution
This repository includes the file `code/fqs.py`, which is **third-party code** and **not original code** written by V. Vijendran.

**Original source:**
> NKrvavica, *fqs: Fast Quartic and Cubic Solver*
> https://github.com/NKrvasica/fqs

Please retain this attribution if redistributing or reusing that file.

## Citation

If you find this repository useful for your research or benchmarking, please cite the associated paper:

```bibtex
@article{vijendran2025nearoptimal,
  title={Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models},
  author={Vijendran, V and Koh, Dax Enshan and Bae, Eunok and Kwon, Hyukjoon and Lam, Ping Koy and Assad, Syed M},
  journal={arXiv preprint arXiv:2501.16419},
  year={2025}
}
```

## License

This project is released under the **MIT License**, which permits use, modification, and distribution with attribution.

```
MIT License © 2026 V. Vijendran
```