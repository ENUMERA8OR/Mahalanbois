import os
import json
import logging
import numpy as np
import arxiv
import lancedb
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# ML & Stats
from sentence_transformers import SentenceTransformer
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import shortest_path

# UI & CLI
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

# Transformers for Local SLM (NovAScore)
import ollama

# Setup Logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mahalanbois_7.5V")

@dataclass
class Paper:
    title: str
    summary: str
    url: str
    published: str
    authors: List[str]
    vector: Optional[np.ndarray] = None
    
    # 2025 Metrics
    mahalanobis_plus: float = 0.0
    ma_dpr: float = 0.0
    suplid: float = 0.0
    nova_score: float = 0.0
    
    # Final Aggregate
    hybrid_novelty: float = 0.0

    def to_dict(self):
        d = asdict(self)
        if d['vector'] is not None:
            d['vector'] = d['vector'].tolist()
        return d

class LanceStore:
    """LanceDB wrapper for high-fidelity scientific vector storage."""
    def __init__(self, db_path: str = ".mahalanbois_db"):
        self.db = lancedb.connect(db_path)
        self.table_name = "papers_v75"

    def save_papers(self, papers: List[Paper]):
        data = [p.to_dict() for p in papers]
        if self.table_name in self.db.list_tables():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data=data)

    def get_history(self, limit: int = 2000) -> List[Paper]:
        if self.table_name not in self.db.list_tables():
            return []
        table = self.db.open_table(self.table_name)
        df = table.to_pandas()
        papers = []
        for _, row in df.head(limit).iterrows():
            p = Paper(
                title=row['title'],
                summary=row['summary'],
                url=row['url'],
                published=row['published'],
                authors=row['authors'],
                vector=np.array(row['vector']) if 'vector' in row else None
            )
            papers.append(p)
        return papers

class NomicEmbedder:
    """High-context scientific embedding using Nomic-Embed-Text-v1.5."""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        logger.info(f"Loading Nomic v1.5 Model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        # Nomic v1.5 recommended prefixes
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        return self.model.encode(prefixed_texts, convert_to_numpy=True)

class GeometricFilter:
    """Mahalanobis++ (2025): Improved OOD detection via L2 Feature Normalization."""
    def __init__(self):
        self.mu = None
        self.precision = None

    def fit(self, X: np.ndarray):
        # Apply L2 Normalization as per Mahalanobis++ (Mueller & Hein, 2025)
        X_norm = normalize(X, norm='l2')
        self.mu = np.mean(X_norm, axis=0)
        
        lw = LedoitWolf()
        lw.fit(X_norm)
        self.precision = lw.precision_

    def score(self, X: np.ndarray) -> np.ndarray:
        X_norm = normalize(X, norm='l2')
        diff = X_norm - self.mu
        # (z - mu)^T Precision (z - mu)
        scores = np.sum(diff @ self.precision * diff, axis=1)
        return scores

class ConnectivityFilter:
    """MA-DPR (2025): Manifold-aware distance via graph connectivity."""
    def __init__(self, k: int = 15):
        self.k = k
        self.X_ref = None

    def fit(self, X: np.ndarray):
        self.X_ref = X

    def score(self, X_query: np.ndarray) -> np.ndarray:
        # Build combined graph for manifold exploration
        combined = np.vstack([self.X_ref, X_query])
        n_ref = len(self.X_ref)
        n_query = len(X_query)
        
        nbrs = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(combined)
        adj = nbrs.kneighbors_graph(combined, mode="distance")
        
        # Compute shortest paths from query points to all points
        dist_matrix = shortest_path(adj, method='auto', directed=False)
        
        # MA-DPR score: Average shortest path to the reference set
        scores = []
        for i in range(n_query):
            query_idx = n_ref + i
            paths_to_ref = dist_matrix[query_idx, :n_ref]
            # Filter infinities (disconnected components)
            valid_paths = paths_to_ref[np.isfinite(paths_to_ref)]
            if len(valid_paths) > 0:
                scores.append(np.mean(valid_paths))
            else:
                scores.append(10.0) # Penalty for total isolation
        
        return np.array(scores)

class ManifoldFilter:
    """SupLID (2025): Local Intrinsic Dimensionality (LID) for manifold-rule violation."""
    def __init__(self, k: int = 20):
        self.k = k
        self.nbrs = None

    def fit(self, X: np.ndarray):
        self.nbrs = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        dists, _ = self.nbrs.kneighbors(X)
        # LID MLE estimator: - (1/k * sum(log(r_i / r_k)))^-1
        # dists[:, -1] is r_k
        rk = dists[:, -1].reshape(-1, 1)
        # Avoid log(0)
        r_i_rk = dists[:, :-1] / (rk + 1e-10)
        log_r = np.log(r_i_rk + 1e-10)
        
        avg_log = np.mean(log_r, axis=1)
        lids = -1.0 / (avg_log + 1e-10)
        return lids

class FactFilter:
    """NovAScore (2025): Atomic Fact Novelty via SLM (Qwen 3)."""
    def __init__(self, model_name: str = "qwen3:0.6b"):
        self.model_name = model_name

    def extract_atomic_facts(self, text: str) -> List[str]:
        prompt = f"""[ATOMIC FACT EXTRACTION]
Extract exactly 5 unique atomic facts (short, independent claims) from the following abstract.
Abstract: {text}
Output as a simple JSON list of strings.
Example: ["Method X uses Y", "Dataset Z was used", ...]"""
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            # Simple extraction from JSON-like block
            if "[" in content and "]" in content:
                json_str = content[content.find("["):content.rfind("]")+1]
                return json.loads(json_str)
            return [line.strip("- ") for line in content.split("\n") if len(line) > 10][:5]
        except:
            return []

    def compute_novelty(self, candidate_facts: List[str], history_embeddings: np.ndarray, embedder: NomicEmbedder) -> float:
        if not candidate_facts: return 0.0
        
        fact_vectors = embedder.embed(candidate_facts, is_query=True)
        # Cross-reference with history
        # Score = Average distance to nearest neighbor in history
        nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(history_embeddings)
        distances, _ = nbrs.kneighbors(fact_vectors)
        
        # High distance = high novelty
        return np.mean(distances)

class Mahalanbois75V:
    """The Ultimate 2025 Novelty Detection Pipeline."""
    def __init__(self):
        self.store = LanceStore()
        self.embedder = NomicEmbedder()
        self.geo_filter = GeometricFilter()
        self.conn_filter = ConnectivityFilter()
        self.manifold_filter = ManifoldFilter()
        self.fact_filter = FactFilter()

    def run(self, query: str, max_history: int = 1000, top_k: int = 5):
        console.print(Panel("[bold cyan]Mahalanbois 7.5V[/bold cyan]\n[dim]High-Fidelity Scientific Novelty Detection (2025 Edition)[/dim]", box=box.DOUBLE))

        # 1. Prepare History
        history = self.store.get_history(limit=max_history)
        if not history:
            logger.info("Database empty. Ingesting baseline papers...")
            self.ingest_baseline("cat:cs.AI OR cat:cs.LG")
            history = self.store.get_history(limit=max_history)

        X_hist = np.array([p.vector for p in history])
        
        # 2. Fit Statistical Filters
        with console.status("[bold blue]Fitting 2025 Metric Filters..."):
            self.geo_filter.fit(X_hist)
            self.conn_filter.fit(X_hist)
            self.manifold_filter.fit(X_hist)

        # 3. Fetch Query Papers
        logger.info(f"Scanning for weak signals: [bold yellow]{query}[/bold yellow]")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=30, sort_by=arxiv.SortCriterion.SubmittedDate)
        
        candidates = []
        for res in client.results(search):
            p = Paper(
                title=res.title,
                summary=res.summary,
                url=res.entry_id,
                published=res.published.isoformat(),
                authors=[a.name for a in res.authors]
            )
            candidates.append(p)
        
        X_query = self.embedder.embed([p.summary for p in candidates], is_query=True)
        for i, p in enumerate(candidates):
            p.vector = X_query[i]

        # 4. Phase 1: Statistical Screening (Mahalanobis++, MA-DPR, SupLID)
        m_plus = self.geo_filter.score(X_query)
        ma_dpr = self.conn_filter.score(X_query)
        suplid = self.manifold_filter.score(X_query)
        
        for i, p in enumerate(candidates):
            p.mahalanobis_plus = m_plus[i]
            p.ma_dpr = ma_dpr[i]
            p.suplid = suplid[i]
            
            # Normalize scores for initial hybrid ranking
            # (Crude normalization for prototype)
            p.hybrid_novelty = (p.mahalanobis_plus * 0.4 + p.ma_dpr * 0.3 + p.suplid * 0.3)

        # 5. Phase 2: Atomic Fact Filtering (NovAScore) for Top Candidates
        candidates = sorted(candidates, key=lambda x: x.hybrid_novelty, reverse=True)
        top_candidates = candidates[:10]
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("[magenta]Running NovAScore (Atomic Fact Analysis)...", total=len(top_candidates))
            for p in top_candidates:
                facts = self.fact_filter.extract_atomic_facts(p.summary)
                p.nova_score = self.fact_filter.compute_novelty(facts, X_hist, self.embedder)
                # Re-calculate hybrid including NovAScore
                p.hybrid_novelty = (p.mahalanobis_plus * 0.25 + 
                                   p.ma_dpr * 0.2 + 
                                   p.suplid * 0.2 + 
                                   p.nova_score * 0.35)
                progress.advance(task)

        # 6. Final Report
        results = sorted(top_candidates, key=lambda x: x.hybrid_novelty, reverse=True)[:top_k]
        
        table = Table(title="🔥 2025 NOVELTY DETECTION REPORT 🔥", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Rank", justify="center")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("M++", style="dim")
        table.add_column("MA-DPR", style="dim")
        table.add_column("SupLID", style="dim")
        table.add_column("NovA", style="yellow")
        table.add_column("Title", style="white")

        for i, p in enumerate(results):
            table.add_row(
                str(i+1),
                f"{p.hybrid_novelty:.3f}",
                f"{p.mahalanobis_plus:.1f}",
                f"{p.ma_dpr:.2f}",
                f"{p.suplid:.1f}",
                f"{p.nova_score:.3f}",
                p.title
            )
        
        console.print(table)
        
        # Save to history for future baseline enhancement
        self.store.save_papers(results)

    def ingest_baseline(self, query: str):
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=300, sort_by=arxiv.SortCriterion.Relevance)
        papers = []
        texts = []
        for res in client.results(search):
            p = Paper(
                title=res.title,
                summary=res.summary,
                url=res.entry_id,
                published=res.published.isoformat(),
                authors=[a.name for a in res.authors]
            )
            papers.append(p)
            texts.append(p.summary)
        
        vectors = self.embedder.embed(texts)
        for p, v in zip(papers, vectors):
            p.vector = v
        self.store.save_papers(papers)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Quantum Neural Networks")
    parser.add_argument("--history", type=int, default=1000)
    args = parser.parse_args()
    
    detector = Mahalanbois75V()
    detector.run(args.query, max_history=args.history)
