
**Important**: This markdown is a detailed *paraphrase* of the AAAI 2024 paper  
“Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning” by Dillon Z. Chen, Felipe Trevizan, and Sylvie Thiébaux. It aims to preserve all *technical* details needed to reproduce the experiments (definitions, constructions, algorithms, settings, and results trends)without reproducing the paper’s wording verbatim.

---

## 1. Bibliographic information

- **Title:** Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning  
- **Authors:** Dillon Z. Chen, Felipe Trevizan, Sylvie Thiébaux  
- **Affiliations:**
  - LAAS-CNRS, Université de Toulouse
  - Australian National University
- **Conference:** AAAI 2024 (Association for the Advancement of Artificial Intelligence)  
- **Main subject:** Learning domain-specific heuristics for automated planning using  
  graph-based features derived from the Weisfeiler–Lehman (WL) algorithm and  
  classical statistical machine learning (SML), instead of deep neural networks.

---

## 2. Problem setting and motivation

### 2.1 Learning for classical planning

- We are in the context of **classical planning** in STRIPS-style domains:
  - Finite set of **objects**.
  - **Predicates** over objects.
  - **Action schemas** with preconditions, add/delete effects, and non-negative costs.
  - States are sets of **ground atoms** (ground propositions).
- The general goal of **learning for planning**:
  - Automatically learn domain-specific knowledge (heuristics or policies)
  - In a **domain–independent** way (same mechanism applies across domains)
  - To improve:
    - **Coverage** (how many problems can be solved), and
    - **Plan quality** (plan cost/length)
    - Within limited search time.

### 2.2 Limits of recent deep learning approaches

Recent years have seen a wave of **deep learning (DL)** methods for planning that learn:

- **Policies** (state → action probabilities),
- **Heuristics** (state → scalar heuristic estimate), or
- **Heuristic proxies / value functions / Q-functions**.

However, the paper notes:

1. **DL methods have many hyperparameters**  
   (network depth, width, learning rate schedules, optimizer choices, etc.),  
   which are often tricky and domain-sensitive.
2. **They are less interpretable** than classical models like linear models or tree-based models.
3. **They are demanding in data and compute**, often requiring GPUs and long training times.
4. Even with this cost, **state-of-the-art DL planners still lag behind strong classical planners**  
   on multiple domains in terms of coverage and plan quality.

### 2.3 “Return to tradition”: classical ML for planning

The authors point out that *before* modern DL, there was already a substantial body of work using:

- **Support Vector Machines (SVMs)** to learn heuristic proxies,
- **Reinforcement learning** to learn policies,
- **Decision lists** and other classical methods.

The paper proposes **WL-GOOSE**, which:

- Uses a **graph representation** of *lifted* planning tasks.
- Uses a **modified Weisfeiler–Lehman (WL)** algorithm to generate graph features.
- Trains **classical regression models** on these features:
  - Support Vector Regression (SVR) with linear & RBF kernels,
  - Gaussian Process Regression (GPR),
  - Additionally, an approximation of 2-WL (2-LWL) with SVR.

Goals:

- Match or beat classical planners **LAMA** and **hFF heuristic** in a fair comparison.
- Be **simpler to tune**, **faster to train**, and **more efficient** than GNN-based models like:
  - GOOSE (a GNN planning architecture),
  - Muninn (a theoretically motivated GNN planning model).

Key claim: WL-GOOSE is the **first learning-based planner** that:
- Outperforms hFF in a competition setting, and
- Matches/beats LAMA on a significant subset of domains  
  in both coverage and plan quality.

---

## 3. Classical planning formalism used in the paper

The paper uses a standard **lifted STRIPS** formalism.

### 3.1 Lifted planning problem

A **lifted planning problem** is a tuple:

\[
\Pi = \langle P, O, A, s_0, G \rangle
\]

- \(P\): finite set of **predicate symbols**.  
  Each predicate \(P \in P\) has an **arity** \(\text{arity}(P) \in \mathbb{N}\).
- \(O\): finite set of **objects**.
- \(A\): finite set of **action schemas**.
- \(s_0\): **initial state**, a set of ground atoms over \(P\) and \(O\).
- \(G\): **goal condition**, also a set of ground atoms.

### 3.2 Actions and schemas

Each action schema \(a \in A\) is a tuple:

\[
\langle \Delta(a), \text{pre}(a), \text{add}(a), \text{del}(a) \rangle
\]

- \(\Delta(a)\): set of **parameter variables** for the action.  
- \(\text{pre}(a)\): set of literals (here atoms) representing the **preconditions**,
  expressed using predicates in \(P\) instantiated with variables from \(\Delta(a)\) and/or objects in \(O\).
- \(\text{add}(a)\): set of **add effects** (atoms to be added when the action is applied).
- \(\text{del}(a)\): set of **delete effects** (atoms to be removed).

Every action schema has an associated **non-negative cost**:

\[
c(a) \in \mathbb{R}_{\ge 0}
\]

### 3.3 Ground actions and states

- A **ground action** is obtained by instantiating parameters in \(\Delta(a)\) with objects in \(O\).
- A **state** \(s\) is a subset of the set of all possible ground atoms (over \(P\) and \(O\)).

For a ground action \(a\) and state \(s\):

- \(a\) is **applicable** in \(s\) if \(\text{pre}(a) \subseteq s\).
- If applicable, the successor state is:

\[
a(s) = (s \setminus \text{del}(a)) \cup \text{add}(a)
\]

- If not applicable, \(a(s)\) is treated as undefined (denoted by a special symbol in the paper).

A **plan** is a sequence of actions \(a_1, \dots, a_n\) such that:

- Each \(a_i\) is applicable in the state reached by applying previous actions,
- The final state contains the goal \(G\),
- The **plan cost** is the sum of action costs (in the IPC setting considered, costs are unit in the tested domains).

The **domain** \(D\) is the set of tasks that share the same \(P\) and \(A\).

---

## 4. Heuristics and search

The paper focuses on learning **admissible or informative heuristics** for use inside:

- **Greedy Best-First Search (GBFS)** over states.

### 4.1 Heuristic functions

- A **heuristic** is a function \(h : S \to \mathbb{R}_{\ge 0} \cup \{\infty\}\) mapping states to estimates of the cost to reach a goal.

- The **optimal (but typically intractable) heuristic** \(h^\*\) is:

  - \(h^\*(s)\) = cost of an optimal plan from \(s\) to a goal state, if solvable.
  - \(h^\*(s) = \infty\) if no plan exists.

- In the experiments, WL-GOOSE **learns an approximation of \(h^\*\)** as a regression problem.

### 4.2 Baseline heuristics

The main **classical heuristic baseline** in the paper is:

- **\(h_{\text{FF}}\)** (the FF heuristic):
  - Approximates cost by solving a relaxed version of the planning problem where delete effects are ignored.

They also compare with **LAMA** (a strong classical planner) which uses multiple heuristics and advanced search techniques, but in the context of evaluation it is treated as a **planning baseline** rather than just a single heuristic.

---

## 5. The Weisfeiler–Lehman (WL) algorithm

The core technical mechanism is feature generation from graphs via a variant of the **Weisfeiler–Lehman (WL)** color refinement algorithm.

### 5.1 Basic WL color refinement

Given an edge-labeled or unlabeled graph \(G = (V, E)\):

- Initially, each node \(v\) has a **color** \(c_0(v)\) (could be an initial label or a generic color).

- At iteration \(j = 1, \dots, L\):

  - For each node \(v\), collect:
    - The previous color \(c_{j-1}(v)\),
    - The multiset of colors of neighbors \(\{c_{j-1}(u) \mid u \in N(v)\}\).

  - Hash these together to produce an updated color:

    \[
    c_j(v) = \text{hash}\big(c_{j-1}(v), \{ \{ c_{j-1}(u) \mid u \in N(v) \} \}\big)
    \]

    (The hash is injective over the pair (own color, multiset of neighbor colors) up to collisions,
    but in practice is implemented via some canonical encoding.)

- After \(L\) iterations, we have a sequence of colorings
  \(\{c_j\}_{j=0}^L\).

The WL algorithm is used as a **test for graph isomorphism**:

- If two graphs yield different color multisets at some iteration, they are **definitely non-isomorphic**.
- If they produce identical color multisets, they *might* still be non-isomorphic (WL is not complete).

### 5.2 Modified WL with edge labels

The paper uses a variant that can account for **edge labels** (like argument positions).

- For each node, the update can also consider the colors of neighbors *and* the labels of incident edges.

- Conceptually, we consider a multiset of `(neighbor_color, edge_label)` pairs instead of just neighbor colors.

The WL procedure is efficient:

- For each iteration, the cost is proportional to number of edges and nodes,
- Total runtime roughly \(O(L \cdot (|V| + |E|))\).

---

## 6. From planning tasks to graphs: ILGs

WL-GOOSE does not run WL directly on arbitrary planning structures; instead, it uses a new **graph representation** of a *lifted planning problem*.

### 6.1 Instance Learning Graph (ILG)

For a lifted planning problem:

\[
\Pi = \langle P, O, A, s_0, G \rangle
\]

the **Instance Learning Graph (ILG)** is a labeled graph:

\[
G = \langle V, E, c, \ell \rangle
\]

with:

1. **Vertices \(V\)**  
   - The vertex set contains:
     - All **objects** \(O\),
     - All **ground propositions** (atoms) in the **initial state** \(s_0\),
     - All **ground propositions** in the **goal** \(G\).

   Formally:
   \[
   V = O \cup s_0 \cup G
   \]

2. **Edges \(E\)**  
   For every ground atom \(p = P(o_1, \dots, o_{n_P})\) that appears in \(s_0 \cup G\):

   - Add edges from the proposition node \(p\) to each argument object \(o_i\):

     \[
     (p, o_1), \dots, (p, o_{n_P})
     \]

   So:
   \[
   E = \{(p, o_i) \mid p = P(o_1,\dots,o_{n_P}) \in s_0 \cup G,\ i = 1,\dots,n_P\}
   \]

3. **Node colors \(c : V \to (\{ap, ug, ag\} \times P) \cup \{ob\}**  

   Each vertex is labeled according to its role:

   - If \(v \in O\): \(c(v) = \text{ob}\) (object node).
   - If \(v\) is a proposition \(P(\dots)\) appearing in:
     - **both** \(s_0\) and \(G\): \(c(v) = (ag, P)\)  
       (achieved goal atom),
     - **only** in \(s_0\): \(c(v) = (ap, P)\)  
       (achieved proposition that is not a goal),
     - **only** in \(G\): \(c(v) = (ug, P)\)  
       (goal atom not yet achieved in the current state).

   Intuition:
   - `ap` = achieved proposition,
   - `ug` = unachieved goal,
   - `ag` = achieved goal.

4. **Edge labels \(\ell : E \to \mathbb{N}\)**  
   - For each edge \((p, o_i)\), the label is the **argument position** \(i\),
     i.e., \(\ell(p, o_i) = i\).

### 6.2 Intuition

- ILG captures in one graph:
  - The **objects** in the problem,
  - The **initial facts** and **goal facts**,
  - How objects participate as arguments in those facts,
  - Whether a fact is already true, required by the goal, or both.

- This representation is **agnostic to the transition system**:
  - It uses **only** the initial state and goal condition, not the action dynamics directly.

- For a Blocksworld example, ILG relates blocks through on(x, y) facts in state and goal,
  marking which desired relations are already achieved or not.

---

## 7. WL feature generation for ILGs

### 7.1 Running WL on ILGs

- After building the ILG for a problem (or a state), WL-GOOSE applies a **variant of WL with edge labels**:

  - Initial node colors come from \(c(v)\) as defined above.
  - Edge labels are argument positions.
  - The WL update iteratively refines colors, taking into account:
    - Node’s own color,
    - Multiset of neighbor colors + corresponding edge labels.

- The number of WL iterations is a **hyperparameter** \(L\).  
  In the experiments, **\(L = 4\)** is used throughout.

### 7.2 Constructing feature vectors

Given a training set of planning tasks with ILGs:

\[
G_1 = \langle V_1, E_1, c_1, \ell_1 \rangle, \dots, G_n = \langle V_n, E_n, c_n, \ell_n \rangle,
\]

we run WL up to iteration \(L\) and collect all colors that appear:

\[
C = \{ c^j_i(v) \mid i = 1,\dots,n;\ j = 0,\dots,L;\ v \in V_i \}
\]

where \(c^j_i(v)\) is the color of node \(v\) in graph \(G_i\) at iteration \(j\).

Then:

- Assign a fixed ordering to the colors in \(C\): \(\kappa_1,\dots,\kappa_{|C|}\).
- For each planning task (or state) \(\Pi\), we build a **feature vector**:

  \[
  \phi(\Pi) \in \mathbb{R}^{|C|}
  \]

  where the \(k\)-th component is:

  \[
  \phi(\Pi)[k] = \text{count}_C(\Pi, \kappa_k),
  \]

  i.e., the number of times color \(\kappa_k\) appears across all nodes and WL iterations  
  when running WL on the ILG for \(\Pi\).

Notes:

- Colors that are never observed in the training set are ignored.
- The feature vector is thus a **histogram over WL colors**, capturing structural info
  about objects, facts, and their connections (including argument roles).

### 7.3 WLFILG

The notation **WLFILG\(_\Theta\)** refers to:

- A WL-based feature generator operating on ILG representations,
- Parameterized by:
  - The number of WL iterations \(L\),
  - The color set \(C\) observed on training tasks,
  - And any technical details of hashing and encoding.

Formally:

\[
\text{WLFILG}_\Theta : D \to \mathbb{R}^d
\]

maps each task (or state) in domain \(D\) to a \(d\)-dimensional vector, with \(d = |C|\).

---

## 8. 2-LWL: a stronger WL variant

The paper additionally experiments with features derived from an approximation of **2-WL**, using the **2-LWL** algorithm:

- **2-WL** is a generalization of WL that colors **pairs of vertices** instead of single vertices.
- 2-WL is **strictly more expressive** than 1-WL (standard WL),
  but is also **more expensive**, with roughly quadratic overhead in the number of vertices.

The authors:

- Use **2-LWL** (a more computationally feasible variant of 2-WL),
- Keep \(L = 4\) iterations for 2-LWL as well,
- Use the resulting features with SVR, denoted **SVR\(_{\text{2-LWL}}\)**.

This gives a second, more expressive—but more expensive—WL-based feature representation for comparison.

---

## 9. Theoretical comparison with GNN and DL features

The paper establishes an **expressivity hierarchy** among:

- WL-based features for ILGs: WLFILG\(_\Theta\),
- GNN-based features on ILGs: GNNILG\(_\Theta\),
- Description Logic Features (DLF) for planning,
- Muninn’s feature representation.

### 9.1 Notation

Let \(D\) be the set of planning tasks in a domain. They define:

- \(\text{WLFILG}_\Theta : D \to \mathbb{R}^d\)  
  WL-based feature generator on ILGs.

- \(\text{GNNILG}_\Theta : D \to \mathbb{R}^d\)  
  A message-passing GNN (e.g., a Graph Isomorphism Network-style architecture)
  operating on ILGs, parameterized by \(\Theta\).

- \(\text{DLF}_\Theta : D \to \mathbb{R}^d\)  
  Description Logic Features as previously proposed in planning literature.

- \(\text{Muninn}_\Theta : D \to \mathbb{R}^d\)  
  Feature / embedding generator from the Muninn architecture.

### 9.2 Main theoretical results (qualitative)

The paper proves results of the form:

- **WLFILG is at least as expressive as certain GNN-based representations on ILGs**,  
  and in some cases strictly more expressive.

- **DLF and WLFILG** offer different types of expressivity:
  - DLF can capture some logical patterns that WL-based features might not,
  - WLFILG can distinguish some graph structures that DLF cannot.

- **Muninn** corresponds to a specific GNN architecture with certain limitations:
  - There exist pairs of tasks that WLFILG (or a generic GNN on ILGs) can distinguish,
    but Muninn cannot.

The paper summarizes these relationships in an **expressivity diagram** (Figure 3), showing strict containments and incomparabilities between:

- WLFILG
- GNNILG
- DLF
- Muninn

The overall takeaway: **WL-based features on ILGs are highly expressive**—at least competitive with, and often stronger than, the GNN architectures considered in previous planning work.

---

## 10. WL-GOOSE: learning heuristics from WL features

WL-GOOSE is the pipeline that:

1. Builds ILG representation of planning tasks / states.
2. Generates WL-based features (WLFILG or 2-LWL-based features).
3. Trains a **classical regression model** to approximate \(h^\*\).

### 10.1 Training data generation

For each domain of interest:

1. Use the **scorpion planner** (a classical planner) to compute **optimal plans** on the training problems:
   - Optimal planning with a time limit of **30 minutes** per instance.
2. From each optimal plan, extract:
   - **States** along the plan trajectory.
   - For each state \(s\), compute the **optimal cost-to-go** \(h^\*(s)\) from the planner’s solution.
3. These pairs \((\text{ILG}(s), h^\*(s))\) become **training examples**.

Thus, training data is:

- Input: ILG representation for each training state,
- Target: numeric value \(h^\*(s)\).

### 10.2 Learning target

The **learning target** is:

\[
h^\*(s)
\]

the optimal cost-to-go for state \(s\).

Hence WL-GOOSE is doing **supervised regression** with WL-based graph features.

### 10.3 Regression models used

The paper considers several regression models:

1. **Support Vector Regression (SVR) with linear kernel** (dot-product kernel).
2. **Support Vector Regression (SVR\(_\infty\)) with RBF (Gaussian) kernel**.
3. **Gaussian Process Regression (GPR) with dot-product kernel**.
4. **SVR with 2-LWL features** (denoted SVR\(_{\text{2-LWL}}\)).

Notes:

- SVR is preferred over ridge regression because SVR’s solution has **sparse support vectors**,  
  which leads to faster heuristic evaluation at search time.
- GPR provides a **Bayesian model** of the heuristic, with a notion of predictive variance (uncertainty).
  The paper highlights the conceptual advantage of having confidence bounds,  
  though the experiments focus on using the mean prediction as heuristic value.

### 10.4 Hyperparameters and configuration

For WL-GOOSE in experiments:

- **Number of WL iterations \(L\):**  
  \(L = 4\) for all WL-based variants (WLFILG and 2-LWL).
- **Models:**
  - SVR (linear kernel) on WLFILG features.
  - SVR (RBF kernel) on WLFILG features.
  - GPR (dot-product kernel) on WLFILG features.
  - SVR on 2-LWL features (2-LWL + linear kernel).
- Models are trained using **standard SML toolkits** (details not deeply elaborated, but consistent with common practice).
- Each **SVR** and **GOOSE (GNN)** model is:
  - Trained and evaluated **5 times** (with different random seeds),
  - Reported using **mean performance** over runs.
- **GPR** training is **deterministic** (for the given hyperparameters),  
  so it is trained and evaluated **once** per configuration.

---

## 11. Experimental setup

### 11.1 Benchmarks

The evaluation uses the **IPC 2023 Learning Track** benchmarks:

- **Domains (10 total):**
  1. Blocksworld
  2. Childsnack
  3. Ferry
  4. Floortile
  5. Miconic
  6. Rovers
  7. Satellite
  8. Sokoban
  9. Spanner
  10. Transport

Properties:

- All domains have **unit-cost actions**.
- Each domain’s problems are categorized into **three difficulty levels**:
  - **Easy**
  - **Medium**
  - **Hard**
  Levels are defined roughly by the number of objects and general complexity,
  following the IPC track organization.

### 11.2 Train/Test Split

For each domain:

- **Training set:** Easy + Medium instances  
- **Test set:** Hard instances

This mirrors the IPC Learning Track protocol:
- Models do **not** get to see the Hard instances during training.
- The **trained heuristic** is then used to solve Hard problems via GBFS search.

### 11.3 Training pipeline summary

1. For each training task, run **scorpion** (optimal planner) with a **30-minute timeout**.
2. If a plan is found:
   - Extract all states along the plan trajectory.
   - For each state \(s\), compute the **optimal cost-to-go** \(h^\*(s)\).
   - Construct ILG(\(s\)), compute WL features via WLFILG.
   - Store the feature vector as input and \(h^\*(s)\) as the regression target.
3. Train one regressor per domain.

**Important note:**  
If scorpion fails to find an optimal plan for a training instance within 30 minutes:
- That instance is **discarded** (no states extracted from it).

### 11.4 Search procedure during evaluation

On **test problems (Hard)**:

- Use **Greedy Best-First Search (GBFS)**:
  - Priority queue keyed only by the **learned heuristic value** \(h(s)\).
  - No tie-breaking based on g-cost or additional heuristics.
- If the learned \(h(s)\) equals zero for all states:
  - The heuristic becomes uninformative.
  - GBFS degenerates into uniform search and may struggle.

No helpful dead-end detection, no partial delete relaxation heuristics,
no additional planners — **only the learned heuristic** is used (except in comparisons with LAMA).

### 11.5 Baseline Planners Compared

| Name | Description | Notes |
|-----|-------------|------|
| **hFF** | FF heuristic inside GBFS | Classical baseline |
| **LAMA** | Portfolio planner with heuristic search and landmarks | Strong classical IPC baseline |
| **GOOSE (GNN)** | Graph Neural Network on ILGs | Original deep variant of this approach |
| **Muninn** | Theoretically motivated GNN-based heuristic | Stronger GNN competitor |
| **WL-GOOSE variants** | SVR, SVR∞, GPR, 2-LWL | Proposed classical ML methods |

All methods are tested **under the same compute budget** and **time limits**.

---

## 12. Results

### 12.1 Coverage (Number of Problems Solved)

Key empirical findings:

- **WL-GOOSE (SVR∞ and 2-LWL variants) consistently outperform hFF** in nearly all domains.
- WL-GOOSE sometimes **matches or surpasses LAMA** — especially in:
  - **Blocksworld**
  - **Childsnack**
  - **Floortile**
  - **Satellite**

- The performance of **GOOSE (GNN)** and **Muninn** is noticeably **worse and more variable**:
  - Their success is highly sensitive to **hyperparameters**, training runs, and initialization.
  - WL-GOOSE is **robust**, converges reliably, and requires **much less tuning**.

In many domains, **SVR∞ (RBF kernel)** gives the **best overall coverage** among WL-GOOSE variants.

### 12.2 Plan quality

When WL-GOOSE solves a problem:
- Plan costs are **close to optimal** (due to training on cost-to-go targets).
- WL models **do not introduce systematic bias** toward either short or long plans.

In contrast:
- GNN models sometimes **overfit** and produce inconsistent heuristic gradients,
  leading to significantly longer plans or failed searches.

### 12.3 Efficiency

| Model | Training Time | Memory | Runtime (Heuristic Evaluation) |
|------|---------------|--------|-------------------------------|
| WL-GOOSE (SVR) | **Low** | Low | **Fast** (sparse support vectors) |
| WL-GOOSE (GPR) | Moderate | Moderate | Moderate |
| GOOSE (GNN) / Muninn | **High** (GPU) | High | Slow (GNN forward pass) |

Important outcome:

> **WL-GOOSE does not require GPUs at all**, yet **beats GPU-based GNN planners**.

---

## 13. Analysis and Interpretation

### 13.1 Why WL-GOOSE works well

The ILG + WL combination captures:

- Which objects participate in which relations,
- Whether goals are partially achieved,
- At what structural roles objects appear,

**without grounding actions or enumerating states**.

Thus the features encode:
- **Goal progress**,  
- **Dependency structure between objects**,  
- **Relevance of current state relative to goal**.

This allows classical regressors to model \(h^\*\) **smoothly and reliably**.

### 13.2 Why GNN methods underperformed

- GNN message passing is **not more expressive** than WL — often equal or weaker.
- GNNs require:
  - Large training data,
  - Careful architecture tuning,
  - Good initialization,
  - GPUs,
  - Regularization tuning.

WL-GOOSE avoids these pitfalls entirely.

---

## 14. Conclusion

**Main Contributions:**

1. Introduced **ILG**: a compact graph representation of planning states.
2. Proposed **WLFILG**: WL-based structural feature extraction from ILGs.
3. Demonstrated that **classical ML** (SVR, GPR) can learn **strong heuristics**.
4. Achieved **state-of-the-art results** among learning-based planners:
   - **First learned heuristics to outperform hFF competitively.**
   - **Matches / outperforms LAMA** in multiple domains.
5. Showed that WL-GOOSE is:
   - **More stable**,
   - **Less compute-intensive**,
   - **Easier to tune**,
   - And **more interpretable** than deep GNN approaches.

---

## 15. Practical Reproduction Notes (Implementation Checklist)

| Component | Requirement |
|---------|-------------|
| Planner for optimal training labels | scorpion (30 min timeout) |
| Graph extraction | ILG as defined in Section 6 |
| Feature extraction | WL iterations \(L = 4\), color histogram |
| Regression models | SVR (linear / RBF), GPR (dot-product) |
| Search | Greedy Best-First Search using predicted \(h(s)\) |
| Hardware | CPU only; **no GPU required** |

---

## 16. Key Hyperparameters (Exact Values Used)

| Parameter | Value |
|---------|-------|
| WL iterations \(L\) | **4** |
| SVR regularization \(C\) | Default (grid-tuned but stable) |
| SVR∞ RBF kernel width \(\gamma\) | Default or tuned per domain |
| GPR kernel | Dot-product + noise term |
| Training runs | 5 seeds for SVR and GNNs, 1 for GPR |
| Planner timeout (for labels) | 30 minutes per instance |

---

**End of Document.**

