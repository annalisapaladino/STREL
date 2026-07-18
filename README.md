# `TorCheck` 🔥✅
A fully-differentiable implementation of Spatio-Temporal Reach and Escape Logic (STREL) semantic trees based on PyTorch

## Install
```console
pip install git+https://github.com/annalisapaladino/STREL.git
```

### STL
You can also only install the Signal Temporal logic (STL) library here:
```console
pip install git+https://github.com/ailab-units/TorCheck.git
```

# TorCheck — Differentiable STREL Semantics in PyTorch

TorCheck is an experimental PyTorch implementation of **Signal Temporal Reach and Escape Logic (STREL)** semantic trees.

The library makes it possible to define spatial, temporal, and spatio-temporal properties as composable Python objects and evaluate them over batched trajectories using either:

- **Boolean semantics**, which states whether a formula is satisfied;
- **quantitative robustness semantics**, which measures the margin by which a formula is satisfied or violated.

Because the quantitative operators are implemented with PyTorch tensor operations, robustness values can be integrated into differentiable learning and optimization pipelines, including policy optimization, trajectory synthesis, and neural-network training.

> **Project status:** research prototype. The repository currently contains the core STREL implementation and two executable examples, but it does not yet include a packaged release, automated tests, or a pinned dependency file.

---

## Contents

- [What the library provides](#what-the-library-provides)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Signal representation](#signal-representation)
- [Core concepts](#core-concepts)
- [Available operators](#available-operators)
- [Quick start](#quick-start)
- [STREL example](#strel-example)
- [Bike-sharing case study](#bike-sharing-case-study)
- [Boolean and quantitative semantics](#boolean-and-quantitative-semantics)
- [Differentiability](#differentiability)
- [Running the examples](#running-the-examples)
- [Current limitations](#current-limitations)
- [Possible applications](#possible-applications)
- [Contributing](#contributing)
- [Citation and acknowledgements](#citation-and-acknowledgements)

---

## What the library provides

TorCheck represents a logical specification as a **semantic tree**. Atomic predicates form the leaves of the tree, while logical, temporal, and spatial operators combine them into increasingly complex formulas.

For example, a specification may express that:

- a signal must remain above a threshold for a time interval;
- a suitable node must exist within a given spatial radius;
- a target node must be reachable through nodes satisfying an intermediate condition;
- a spatial condition must eventually become true;
- a node must be surrounded by nodes satisfying another predicate.

A formula can then be evaluated over all nodes and all available time instants.

The central workflow is:

```text
PyTorch trajectory tensor
        ↓
Atomic predicates
        ↓
Logical / temporal / spatial operators
        ↓
Boolean satisfaction and quantitative robustness
        ↓
Optional gradient-based optimization
```

---

## Repository structure

```text
STREL/
├── README.md
├── example_stl_usage.py
├── example_strel_usage.ipynb
├── bike_sharing.ipynb
└── torcheck/
    ├── __init__.py
    └── strel.py
```

### `torcheck/strel.py`

The main implementation. It contains:

- the abstract `Node` interface;
- atomic predicates;
- Boolean operators;
- temporal operators;
- spatial STREL operators;
- Boolean and quantitative evaluation methods;
- utilities for Euclidean and directional distance matrices.

### `example_stl_usage.py`

A minimal script showing how to construct and evaluate a temporal formula.

### `example_strel_usage.ipynb`

A more complete notebook exercising the STREL operators on a small synthetic graph with moving nodes.

### `bike_sharing.ipynb`

A compact case study in which bike-sharing stations are represented as spatial nodes and their bicycle availability changes over time. The notebook demonstrates temporal, spatial, and combined spatio-temporal specifications.

---

## Installation

### Install directly from GitHub

```bash
pip install git+https://github.com/annalisapaladino/STREL.git
```

### Development installation

Clone the repository:

```bash
git clone https://github.com/annalisapaladino/STREL.git
cd STREL
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the current runtime dependencies:

```bash
pip install torch numpy matplotlib jupyter
```

The implementation is based primarily on PyTorch. NumPy, Matplotlib, and Jupyter are required by the example notebooks.

### STL-only dependency

The original TorCheck STL implementation can be installed separately with:

```bash
pip install git+https://github.com/ailab-units/TorCheck.git
```

---

## Signal representation

The most important requirement when using the library is the shape and meaning of the input tensor.

### STREL tensor shape

Spatial and spatio-temporal operators expect a four-dimensional PyTorch tensor:

```text
[B, N, F, T]
```

where:

| Dimension | Meaning |
|---|---|
| `B` | batch size or number of independent trajectories |
| `N` | number of spatial nodes or agents |
| `F` | number of features associated with each node |
| `T` | number of discrete time steps |

The first two features are expected to contain the spatial coordinates:

```text
feature 0 = x-coordinate
feature 1 = y-coordinate
```

Some directional distance functions also use velocity components:

```text
feature 2 = x-velocity
feature 3 = y-velocity
```

Additional state variables follow after the positional or velocity features. For example:

```text
[x, y, vx, vy, battery]
```

or, in the bike-sharing example:

```text
[x, y, vx, vy, number_of_bikes]
```

### Pure STL tensor shape

The original STL-style interface is documented in parts of the code using:

```text
[B, F, T]
```

The current STREL examples use the node-aware four-dimensional representation. When building a new example, use `[B, N, F, T]` unless the selected operator explicitly expects the simpler STL layout.

### Example tensor

```python
import torch

B, N, F, T = 2, 5, 3, 20
signal = torch.zeros((B, N, F, T), dtype=torch.float32)

# Static node coordinates
signal[:, :, 0, :] = torch.rand(B, N, 1)
signal[:, :, 1, :] = torch.rand(B, N, 1)

# Dynamic scalar feature
signal[:, :, 2, :] = torch.randn(B, N, T)
```

---

## Core concepts

### Atomic predicates

An `Atom` compares one selected feature with a scalar threshold.

```python
Atom(var_index, threshold, lte=False)
```

Parameters:

- `var_index`: index of the feature to evaluate;
- `threshold`: comparison threshold;
- `lte=False`: predicate is interpreted as a greater-than condition;
- `lte=True`: predicate is interpreted as a less-than-or-equal condition.

Example:

```python
from torcheck.strel import Atom

high_value = Atom(var_index=2, threshold=3.0, lte=False)
low_value = Atom(var_index=2, threshold=1.0, lte=True)
```

Conceptually:

```text
high_value: feature[2] > 3.0
low_value:  feature[2] <= 1.0
```

### Semantic trees

Operators receive one or more child formulas and return a new formula:

```python
safe = Atom(var_index=2, threshold=0.0, lte=False)
goal = Atom(var_index=2, threshold=5.0, lte=False)

formula = Eventually(And(safe, goal), left_time_bound=0, right_time_bound=4)
```

This compositional design allows specifications to be built directly in Python.

---

## Available operators

### Logical operators

| Operator | Constructor | Meaning |
|---|---|---|
| Negation | `Not(child)` | the child formula must not hold |
| Conjunction | `And(left_child, right_child)` | both formulas must hold |
| Disjunction | `Or(left_child, right_child)` | at least one formula must hold |

### Temporal operators

| Operator | Constructor | Informal meaning |
|---|---|---|
| Globally | `Globally(child, ...)` | the property holds throughout a temporal interval |
| Eventually | `Eventually(child, ...)` | the property becomes true within a temporal interval |
| Until | `Until(left_child, right_child, ...)` | the left property holds until the right one becomes true |
| Since | `Since(left_child, right_child, ...)` | past-time counterpart of `Until` |

Bounded example:

```python
phi = Eventually(
    child=goal,
    left_time_bound=0,
    right_time_bound=5,
)
```

Unbounded example:

```python
phi = Globally(
    child=safe,
    unbound=True,
    right_time_bound=4,
)
```

### Spatial operators

| Operator | Constructor | Informal meaning |
|---|---|---|
| Somewhere | `Somewhere(child, d2, ...)` | a node satisfying `child` exists within distance `d2` |
| Everywhere | `Everywhere(child, d2, ...)` | all relevant nodes within distance `d2` satisfy `child` |
| Reach | `Reach(left_child, right_child, d1, d2, ...)` | a `right_child` node is spatially reachable through nodes satisfying `left_child` |
| Escape | `Escape(child, d1, d2, ...)` | the property holds along an escaping spatial path within the distance bounds |
| Surround | `Surround(left_child, right_child, d2, ...)` | a region satisfying the left formula is spatially enclosed by the right formula |

The available spatial operators use node positions to derive pairwise distances. By default, Euclidean distance is used.

---

## Quick start

The following example creates a small graph of four spatial nodes. Each node has an `x` position, a `y` position, and one scalar state feature.

The formula states:

> Within the next three time steps, there must be a node no farther than 1.5 spatial units whose scalar feature is greater than 0.5.

```python
import torch

from torcheck.strel import Atom, Eventually, Somewhere

B, N, F, T = 1, 4, 3, 6
signal = torch.zeros((B, N, F, T), dtype=torch.float32)

# Coordinates
signal[0, 0, 0:2, :] = torch.tensor([[0.0], [0.0]])
signal[0, 1, 0:2, :] = torch.tensor([[1.0], [0.0]])
signal[0, 2, 0:2, :] = torch.tensor([[0.0], [1.0]])
signal[0, 3, 0:2, :] = torch.tensor([[2.0], [2.0]])

# Scalar feature over time
signal[0, :, 2, :] = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.2, 0.3, 0.6, 0.7, 0.7, 0.8],
    [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
])

active = Atom(var_index=2, threshold=0.5, lte=False)
near_active_node = Somewhere(active, d2=1.5)
formula = Eventually(
    near_active_node,
    left_time_bound=0,
    right_time_bound=3,
)

boolean_result = formula.boolean(signal, evaluate_at_all_times=True)
robustness = formula.quantitative(signal, evaluate_at_all_times=True)

print("Boolean semantics:")
print(boolean_result)

print("Quantitative robustness:")
print(robustness)
```

A positive quantitative result indicates satisfaction. A negative result indicates violation.

---

## STREL example

The notebook `example_strel_usage.ipynb` constructs a synthetic scenario containing seven nodes over five time steps.

Each node has:

- a two-dimensional position;
- a scalar activation feature;
- in some tests, time-varying coordinates.

The notebook evaluates the principal spatial operators:

- `Reach`;
- `Escape`;
- `Somewhere`;
- `Everywhere`;
- `Surround`.

It also compares:

- Boolean outputs;
- quantitative robustness outputs;
- gradients obtained from quantitative semantics.

This notebook is the best entry point for understanding how the individual STREL operators behave on a small controlled example.

Open it with:

```bash
jupyter notebook example_strel_usage.ipynb
```

---

## Bike-sharing case study

The notebook `bike_sharing.ipynb` demonstrates how the library can describe a simple urban resource-distribution problem.

### Scenario

The example contains five bike-sharing stations:

- four stations placed at the corners of a square;
- one central station;
- six time steps;
- a changing number of bicycles at every station.

The node feature vector combines:

```text
[x, y, vx, vy, number_of_bikes]
```

The positions remain fixed, while bicycle availability changes over time.

### Temporal cases

The notebook evaluates properties such as:

1. **Eventually enough bicycles are available**

   > Within two time steps, a station will have more than five bicycles.

   ```python
   Eventually(phi_ge5, left_time_bound=0, right_time_bound=2)
   ```

2. **Availability remains adequate**

   > During the next two time steps, a station always has more than three bicycles.

   ```python
   Globally(phi_ge3, left_time_bound=0, right_time_bound=2)
   ```

3. **A station becomes empty**

   > Within four time steps, there is a moment when no bicycles are available.

   ```python
   Eventually(phi_eq0, left_time_bound=0, right_time_bound=4)
   ```

### Spatial cases

The notebook then analyzes spatial availability:

1. a station with at least five bicycles exists within one kilometre;
2. every station within one kilometre has at least three bicycles;
3. a station is surrounded by stations with high availability;
4. a well-stocked station is reachable within 1.5 kilometres;
5. an escape path exists toward stations with sufficient availability.

Example:

```python
phi_space1 = Somewhere(phi_ge5, d2=1.0)
phi_space2 = Everywhere(phi_ge3, d2=1.0)
phi_space4 = Reach(
    left_child=phi_ge3,
    right_child=phi_ge5,
    d1=0.0,
    d2=1.5,
)
```

### Combined spatio-temporal cases

The final section combines temporal and spatial operators. For example:

> Within two time steps, a station with at least five bicycles will exist within one kilometre.

```python
phi = Eventually(
    Somewhere(phi_ge5, d2=1.0),
    left_time_bound=0,
    right_time_bound=2,
)
```

This example shows why STREL is useful: the desired condition depends simultaneously on **what happens**, **where it happens**, and **when it happens**.

> The current notebook imports `from strel import *`. When running it from the repository root, change this to `from torcheck.strel import *` unless the module has been installed or exposed separately in the active Python environment.

---

## Boolean and quantitative semantics

Every formula derives from `Node` and exposes two principal evaluation methods.

### Boolean semantics

```python
result = formula.boolean(
    signal,
    evaluate_at_all_times=True,
)
```

This returns the truth value of the formula for the available nodes and time instants.

Use Boolean semantics when the main question is simply:

```text
Is the specification satisfied?
```

### Quantitative robustness

```python
robustness = formula.quantitative(
    signal,
    normalize=False,
    evaluate_at_all_times=True,
)
```

Robustness adds a margin:

- `robustness > 0`: the formula is satisfied;
- `robustness < 0`: the formula is violated;
- `robustness ≈ 0`: the signal lies near the satisfaction boundary;
- a larger positive value generally indicates stronger satisfaction;
- a more negative value generally indicates a stronger violation.

This distinction is valuable in optimization. Two trajectories may both satisfy a formula, but the one with the larger robustness has a greater safety or performance margin.

### Evaluation only at the initial time

Set:

```python
evaluate_at_all_times=False
```

when only the semantics at `t = 0` are required.

---

## Differentiability

The quantitative semantics can participate in a PyTorch computational graph.

A typical optimization workflow is:

```python
signal = signal.clone().detach().requires_grad_(True)

robustness = formula.quantitative(
    signal,
    evaluate_at_all_times=True,
)

loss = -robustness.mean()
loss.backward()

print(signal.grad)
```

This allows robustness to be used as:

- a differentiable loss;
- a reward or objective for policy optimization;
- a constraint-related regularizer;
- a trajectory-quality signal;
- a component in model-based control.

Boolean semantics should not be expected to provide meaningful gradients because hard truth values are inherently non-differentiable.

---

## Running the examples

### Minimal STL script

```bash
python example_stl_usage.py
```

The script currently selects CUDA explicitly:

```python
device = torch.device("cuda")
```

For compatibility with machines without a CUDA-enabled GPU, replace it with:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Also note that the example imports `torcheck.stl`, while this repository contains `torcheck/strel.py`. For the present repository, the import should be adapted to the module actually available.

### STREL notebook

```bash
jupyter notebook example_strel_usage.ipynb
```

### Bike-sharing notebook

```bash
jupyter notebook bike_sharing.ipynb
```

Alternatively:

```bash
jupyter lab
```

and open the desired notebook from the browser interface.

---

## Current limitations

This repository is useful as a research implementation, but several aspects should be treated carefully before production use.

### 1. Research-prototype packaging

The repository does not currently expose a complete standard Python packaging configuration in the root directory. Depending on the environment, installation directly through `pip` may require adding a `pyproject.toml` or `setup.py` file.

### 2. No pinned dependency versions

There is no `requirements.txt` or lock file. Reproducibility across PyTorch versions is therefore not guaranteed.

### 3. Import inconsistencies in examples

The examples refer to different module names:

- `torcheck.stl` in `example_stl_usage.py`;
- `torcheck.strel` in `example_strel_usage.ipynb`;
- `strel` in `bike_sharing.ipynb`.

The implementation present in this repository is:

```python
from torcheck.strel import ...
```

The examples should be normalized accordingly.

### 4. Explicit distance loops

Some distance-matrix utilities use explicit loops over batches, nodes, and time steps. This is easy to inspect but may become slow for large graphs, long trajectories, or large batches.

### 5. Device handling

Some tensors are created without explicitly inheriting the device and data type of the input tensor. Additional work may be required for complete and reliable GPU execution.

### 6. Limited validation and error messages

The current implementation assumes that tensors have the expected feature order and dimensionality. More systematic shape validation would make the API safer.

### 7. No automated test suite

The notebooks provide valuable demonstrations, but they do not replace unit, gradient, numerical, and regression tests.

### 8. Directional-distance utilities require review

The code contains directional distance helpers for front, back, left, and right relations. These utilities should be validated carefully before use, particularly in safety-critical applications.

### 9. Scalability

The current MLP- and dense-distance-oriented approach is best suited to relatively small fixed-size environments. Large dynamic graphs may require vectorized distance computation, sparse adjacency structures, or graph-neural-network integration.

---

## Possible applications

The library can be used as a foundation for experiments involving:

- multi-agent robotics;
- drone and vehicle coordination;
- smart-city infrastructure;
- bike- or car-sharing systems;
- distributed sensor networks;
- traffic and mobility analysis;
- energy-grid monitoring;
- swarm behavior;
- spatial resource allocation;
- formal-specification-guided reinforcement learning;
- differentiable trajectory optimization;
- runtime monitoring of spatial-temporal requirements.

---

## Contributing

Contributions are welcome, especially in the following areas:

- standard Python packaging;
- dependency management;
- consistent import paths;
- vectorized distance computation;
- CUDA-safe tensor creation;
- unit and gradient tests;
- API documentation;
- additional STREL examples;
- support for custom graph topologies and weighted adjacency matrices;
- benchmarks against alternative implementations.

A recommended development workflow is:

```bash
git checkout -b feature/my-improvement
# make and test the changes
git commit -m "Describe the improvement"
git push origin feature/my-improvement
```

Then open a pull request describing:

- the problem addressed;
- the proposed implementation;
- the tests performed;
- any effect on existing semantics or tensor shapes.

---

## Citation and acknowledgements

The implementation is based on the TorCheck project and on research concerning Signal Temporal Logic and Signal Temporal Reach and Escape Logic.

The source file includes acknowledgements to:

- Luca Bortolussi;
- Laura Nenzi;
- the AI-CPS Group at the University of Trieste.
