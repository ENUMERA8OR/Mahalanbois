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

1.  **Mahalanobis++ (Geometric Analysis):** Utilizes L2-normalized feature spaces and Ledoit-Wolf covariance estimation. This ensures robust distance measurements in high-dimensional manifolds where standard Euclidean metrics often fail.
2.  **MA-DPR (Graph Connectivity):** A manifold-aware distance metric that evaluates graph-based connectivity. It identifies "information islands" by calculating the mean shortest-path distance between query points and the historical reference set.
3.  **SupLID (Topological Violation):** Measures **Local Intrinsic Dimensionality (LID)**. By analyzing the expansion rate of neighborhood densities, it detects samples that violate the topological constraints of the established research distribution.
4.  **Structural NovAScore (Information Gain):** A deterministic alternative to LLM-based fact extraction. It calculates sentence-level information gain by identifying "Peak Novelty" claims that exhibit the highest semantic distance from historical baselines.

---

## 🏗️ Architectural Evolution

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

## 📈 Roadmap & History

*   **V1-V3:** Initial experimentation with Mahalanobis distance in latent vector spaces.
*   **V4-V5:** Implementation of LanceDB for persistent historical benchmarking.
*   **V7.5:** Exploration of SLM-based atomic fact filtering (NovAScore).
*   **V7.6:** Optimization into a fully deterministic topological pipeline for high-throughput production environments.

---

## ⚖️ License

This project is released under the MIT License.
