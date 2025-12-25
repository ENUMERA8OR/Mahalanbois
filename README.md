# MAHALANBOIS: SEE SOONER, ACT FASTER.

**Mahalanbois** is a deterministic framework for Out-of-Distribution (OOD) detection and novelty analysis in scientific literature. By leveraging high-dimensional topological analysis and manifold learning, it identifies significant research shifts within ArXiv data streams without the computational overhead or stochastic nature of Large Language Models.

---

## 🎯 Value Proposition

The project addresses the "Semantic Signal" problem in research: how to identify truly novel contributions amidst a high volume of incremental publications.

*   **SEE SOONER:** Detect architectural and conceptual anomalies in research manifolds before they reach mainstream citation benchmarks.
*   **ACT FASTER:** Achieve high-throughput analysis (sub-100ms per entry) using a CPU-optimized, deterministic pipeline that eliminates generative latency.

---

## 🔬 Core Methodology (The 7.6V Pipeline)

Mahalanbois 7.6V implements a multi-stage ensemble of community-validated OOD metrics, optimized for high-dimensional embedding spaces:

1.  **Mahalanobis++ (Geometric Analysis):** Utilizes L2-normalized feature spaces and Ledoit-Wolf covariance estimation. 
    *   *Citation:* Mueller & Hein, "Mahalanobis++: Improving OOD Detection via Feature Normalization", [arXiv:2505.18032](https://arxiv.org/abs/2505.18032) (2025).
2.  **MA-DPR (Graph Connectivity):** A manifold-aware distance metric that evaluates graph-based connectivity to identify "information islands."
    *   *Citation:* Liu et al., "MA-DPR: Manifold-aware Distance Metrics for Dense Passage Retrieval", [arXiv:2509.13562](https://arxiv.org/abs/2509.13562) (2025).
3.  **SupLID (Topological Violation):** Measures **Local Intrinsic Dimensionality (LID)** to detect topological constraints violations.
    *   *Citation:* "SupLID: Geometrical Guidance for Out-of-Distribution Detection in Semantic Segmentation", [arXiv:2511.18816](https://arxiv.org/abs/2511.18816) (2025).
4.  **Structural NovAScore (Information Gain):** A deterministic alternative to LLM-based fact extraction focusing on sentence-level novelty.
    *   *Citation:* Ai et al., "NovAScore: A New Automated Metric for Evaluating Document Level Novelty", [arXiv:2409.09249](https://arxiv.org/abs/2409.09249) (2024).

---

## 🔄 Execution Flow: Input to Output

Mahalanbois 7.6V operates as a sequential pipeline designed for maximum precision and minimal latency:

1.  **Ingestion & Vectorization (Input):**
    *   The script takes a search query (e.g., via ArXiv API) and fetches the latest document abstracts.
    *   Each document is passed through the **Nomic-Embed-Text-v1.5** model to generate a high-dimensional vector representation.

2.  **Baseline Benchmarking:**
    *   Historical data is retrieved from **LanceDB**.
    *   The script fits the **Geometric (Mahalanobis++)**, **Graph (MA-DPR)**, and **Manifold (SupLID)** filters against the historical distribution to establish a "normative" manifold.

3.  **Topological Screening:**
    *   Query vectors are scored against the baseline filters. This identifies global anomalies (Far-OOD) and manifold-rule violations (Near-OOD).

4.  **Granular Semantic Analysis:**
    *   Top candidates from the screening phase are decomposed into individual sentences.
    *   Each sentence is vectorized and scored for **Structural NovAScore (Information Gain)** against the history to pinpoint the exact claim that represents a "Peak Novelty" event.

5.  **Hybrid Synthesis & Reporting (Output):**
    *   All scores are normalized and fused into a final **Hybrid Novelty Score**.
    *   The **Rich CLI** generates a high-density report ranking the papers, allowing the user to "See Sooner" and identify groundbreaking research instantly.

---

## 💎 The Mathematical Framework of Mahalanbois 7.6V

The project evolved from a generative-heavy approach to a pure topological framework:

| Metric | Version 7.5V (Generative) | Version 7.6V (Deterministic) |
| :--- | :--- | :--- |
| **Orchestration** | SLM-driven (Qwen3-0.6b) | **Manifold-driven (Matrix Math)** |
| **Latency** | 10s - 30s per abstract | **< 100ms per abstract** |
| **Consistency** | Stochastic / Non-deterministic | **100% Mathematically Reproducible** |
| **Requirements** | VRAM-intensive (GPU required) | **Low-footprint (CPU optimized)** |

---

## 🛠️ Technical Implementation

### Prerequisites
*   **Python 3.9+**
*   **Vector Engine:** [LanceDB](https://lancedb.com/) for low-latency, disk-based vector persistence.
*   **Embeddings:** [Nomic-Embed-Text-v1.5](https://nomic.ai/) (Matryoshka-capable, 8192 context length).

### Installation
```bash
pip install -r requirements.txt
```

### Execution
To analyze a specific research domain for structural novelty:
```bash
python mahalanbois7.6V.py --query "category-agnostic novelty detection"
```

---

## 💎 The Mathematical Framework of Mahalanbois 7.6V

Mahalanbois 7.6V treats research not as text, but as a **Topological Manifold**—a curved surface of human knowledge in high-dimensional space. The pipeline identifies novelty by measuring how a new data point "warps" this surface through four distinct lenses:

### 1. Geometric Analysis: Mahalanobis++
Standard Euclidean distance is isotropic; it fails to account for the "shape" of data distributions.
*   **Intuition:** Mahalanobis++ scales the feature space based on the density and covariance of existing knowledge.
*   **The Math:** By applying **L2 Normalization**, we project research onto a unit hypersphere, focusing purely on semantic direction. We utilize **Ledoit-Wolf Shrinkage** for high-dimensional stability, ensuring the precision matrix remains well-conditioned even with limited historical samples.

### 2. Connectivity Analysis: MA-DPR
Proximity in latent space does not always imply intellectual continuity.
*   **Intuition:** Imagine a knowledge graph where papers are connected by conceptual "roads."
*   **The Math:** MA-DPR calculates the **Shortest Path** from a new entry to the historical "mainland." If a paper is physically close but lacks a high-probability path through the manifold, it is flagged as an **Information Island**—a disconnected conceptual leap.

### 3. Topological Violation: SupLID
Every research domain possesses a characteristic **Intrinsic Dimensionality**.
*   **Intuition:** A point that forces a 2D plane to behave like a 3D volume is a structural anomaly.
*   **The Math:** **Local Intrinsic Dimensionality (LID)** evaluates the expansion rate of neighborhood densities. A novel paper often exhibits a dimensionality spike, indicating it sits in a region of the manifold that violates the established topological constraints.

### 4. Semantic Information Gain: Structural NovAScore
Global averaging often dilutes the signal of groundbreaking innovation.
*   **Intuition:** A novel paper often contains 90% established context and one or two radical, unprecedented claims.
*   **The Math:** We decompose abstracts into constituent sentence vectors and analyze the **Peak Novelty** (90th percentile distance). This identifies the specific "information spark" hidden within standard academic prose, rather than averaging out the innovation.

---

## 🚀 Future Directions: The OOD-38 Diagnostic Suite

The next evolution of Mahalanbois involves transitioning from a novelty filter to a comprehensive **OOD Diagnostic Suite**.

*   **Metric Unification:** Integration of the 38 community-validated OOD metrics (Energy-based, ReAct, VIM, DICE, etc.) to create a high-dimensional "Anomalous Signature."
*   **Neural Orchestration:** Utilizing an **SLM (Small Language Model)** like Qwen3-0.6b as a "Neural Fuser." Instead of simple weighting, the SLM will interpret the 38-dimensional signature to provide qualitative diagnostic explanations of *why* a piece of research is novel.
*   **Real-time Manifold Monitoring:** Moving toward live-streaming ArXiv integration with automated discovery alerts for high-entropy research signals.

---

## ⚖️ License

This project is released under the MIT License.
