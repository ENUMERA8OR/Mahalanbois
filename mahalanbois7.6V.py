import os
import json
import logging
import re
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

# Setup Logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mahalanbois_7.6")

@dataclass
class Paper:
    title: str
    summary: str
    url: str
    published: str
    authors: List[str]
    vector: Optional[np.ndarray] = None
    
    # 2025 Deterministic Metrics
    mahalanobis_plus: float = 0.0
    ma_dpr: float = 0.0
    suplid: float = 0.0
    structural_nova: float = 0.0
    
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
        self.table_name = "papers_v76"

    def save_papers(self, papers: List[Paper]):
        data = [p.to_dict() for p in papers]
        try:
            table = self.db.open_table(self.table_name)
            table.add(data)
        except Exception:
            # Table probably doesn't exist
            self.db.create_table(self.table_name, data=data)

    def get_history(self, limit: int = 2000) -> List[Paper]:
        try:
            table = self.db.open_table(self.table_name)
        except Exception:
            logger.warning(f"Table {self.table_name} not found.")
            return []
        
        df = table.to_pandas()
        logger.info(f"Loaded {len(df)} rows from {self.table_name}")
        papers = []
        for _, row in df.head(limit).iterrows():
            if 'vector' not in row or row['vector'] is None:
                continue
            p = Paper(
                title=row['title'],
                summary=row['summary'],
                url=row['url'],
                published=row['published'],
                authors=row['authors'],
                vector=np.array(row['vector'])
            )
            papers.append(p)
        logger.info(f"Successfully parsed {len(papers)} papers from history.")
        return papers

class NomicEmbedder:
    """High-context scientific embedding using Nomic-Embed-Text-v1.5."""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        logger.info(f"Loading Nomic v1.5 Model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        return self.model.encode(prefixed_texts, convert_to_numpy=True)

class GeometricFilter:
    """Mahalanobis++ (2025): Improved OOD detection via L2 Feature Normalization."""
    def __init__(self):
        self.mu = None
        self.precision = None

    def fit(self, X: np.ndarray):
        X_norm = normalize(X, norm='l2')
        self.mu = np.mean(X_norm, axis=0)
        lw = LedoitWolf()
        lw.fit(X_norm)
        self.precision = lw.precision_

    def score(self, X: np.ndarray) -> np.ndarray:
        X_norm = normalize(X, norm='l2')
        diff = X_norm - self.mu
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
        combined = np.vstack([self.X_ref, X_query])
        n_ref = len(self.X_ref)
        n_query = len(X_query)
        nbrs = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(combined)
        adj = nbrs.kneighbors_graph(combined, mode="distance")
        dist_matrix = shortest_path(adj, method='auto', directed=False)
        scores = []
        for i in range(n_query):
            query_idx = n_ref + i
            paths_to_ref = dist_matrix[query_idx, :n_ref]
            valid_paths = paths_to_ref[np.isfinite(paths_to_ref)]
            scores.append(np.mean(valid_paths) if len(valid_paths) > 0 else 10.0)
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
        rk = dists[:, -1].reshape(-1, 1)
        r_i_rk = dists[:, :-1] / (rk + 1e-10)
        log_r = np.log(r_i_rk + 1e-10)
        avg_log = np.mean(log_r, axis=1)
        lids = -1.0 / (avg_log + 1e-10)
        return lids

class StructuralFactFilter:
    """Deterministic NovAScore: Sentence-Level Information Gain."""
    def __init__(self):
        # Regex to split text into sentences robustly
        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    def split_sentences(self, text: str) -> List[str]:
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def compute_novelty(self, paper_text: str, history_embeddings: np.ndarray, embedder: NomicEmbedder) -> float:
        sentences = self.split_sentences(paper_text)
        if not sentences: return 0.0
        
        sentence_vectors = embedder.embed(sentences, is_query=True)
        # Use Cosine distance to find the "strangeness" of each claim
        nbrs = NearestNeighbors(n_neighbors=1, metric="cosine").fit(history_embeddings)
        distances, _ = nbrs.kneighbors(sentence_vectors)
        
        # Aggregate logic: The "Peak Novelty" across all sentences
        # This identifies papers that have at least one or two completely new 'facts'
        # compared to papers that just use 'new words' throughout.
        peak_novelty = np.percentile(distances, 90) # Look at top 10% of novel claims
        average_novelty = np.mean(distances)
        
        # Hybrid Info-Gain score
        return (peak_novelty * 0.7 + average_novelty * 0.3)

class Mahalanbois76:
    """The Ultimate Deterministic 2025 Topological Pipeline (No SLM)."""
    def __init__(self):
        self.store = LanceStore()
        self.embedder = NomicEmbedder()
        self.geo_filter = GeometricFilter()
        self.conn_filter = ConnectivityFilter()
        self.manifold_filter = ManifoldFilter()
        self.fact_filter = StructuralFactFilter()

    def run(self, query: str, max_history: int = 1000, top_k: int = 5):
        console.print(Panel("[bold green]Mahalanbois 7.6[/bold green]\n[dim]Pure Topological Novelty Detection (Deterministic & SLM-Free)[/dim]", box=box.DOUBLE))

        # 1. Prepare History
        history = self.store.get_history(limit=max_history)
        if not history:
            logger.info("Database empty. Ingesting baseline...")
            self.ingest_baseline("cat:cs.AI OR cat:cs.LG")
            history = self.store.get_history(limit=max_history)
        
        if not history:
            logger.error("Failed to ingest or load history. Exiting.")
            return

        X_hist = np.array([p.vector for p in history])
        if X_hist.ndim == 1:
            X_hist = np.stack(X_hist)
        
        # 2. Fit Filters
        with console.status("[bold blue]Analyzing Manifold Topology..."):
            self.geo_filter.fit(X_hist)
            self.conn_filter.fit(X_hist)
            self.manifold_filter.fit(X_hist)

        # 3. Fetch Candidates
        logger.info(f"Scanning for structural anomalies: [bold yellow]{query}[/bold yellow]")
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
        if X_query.ndim == 1:
            X_query = np.stack(X_query)
        
        for i, p in enumerate(candidates):
            p.vector = X_query[i]

        # 4. Phase 1: Global & Graph Filtering
        m_plus = self.geo_filter.score(X_query)
        ma_dpr = self.conn_filter.score(X_query)
        suplid = self.manifold_filter.score(X_query)
        
        for i, p in enumerate(candidates):
            p.mahalanobis_plus = m_plus[i]
            p.ma_dpr = ma_dpr[i]
            p.suplid = suplid[i]

        # 5. Phase 2: Structural NovAScore (Sentence-Level Information Gain)
        # This replaces the slow SLM/NovAScore with fast matrix math
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("[green]Calculating Semantic Information Gain...", total=len(candidates))
            for p in candidates:
                p.structural_nova = self.fact_filter.compute_novelty(p.summary, X_hist, self.embedder)
                # Weighted Final Hybrid Score
                p.hybrid_novelty = (p.mahalanobis_plus * 0.2 + 
                                   p.ma_dpr * 0.25 + 
                                   p.suplid * 0.25 + 
                                   p.structural_nova * 50.0) # Scaling factor for comparable weights
                progress.advance(task)

        # 6. Final Report
        results = sorted(candidates, key=lambda x: x.hybrid_novelty, reverse=True)[:top_k]
        
        table = Table(title="💎 PURE TOPOLOGICAL NOVELTY REPORT 💎", box=box.ROUNDED, header_style="bold green")
        table.add_column("Rank", justify="center")
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("M++", style="dim")
        table.add_column("MA-DPR", style="dim")
        table.add_column("SupLID", style="dim")
        table.add_column("Info-Gain", style="yellow")
        table.add_column("Title", style="white")

        for i, p in enumerate(results):
            table.add_row(
                str(i+1),
                f"{p.hybrid_novelty:.3f}",
                f"{p.mahalanobis_plus:.1f}",
                f"{p.ma_dpr:.2f}",
                f"{p.suplid:.1f}",
                f"{p.structural_nova:.3f}",
                p.title
            )
        
        console.print(table)
        self.store.save_papers(results)

    def ingest_baseline(self, query: str):
        logger.info(f"Ingesting baseline for query: {query}")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=200, sort_by=arxiv.SortCriterion.Relevance)
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
        
        if not texts:
            logger.error("No texts found to embed during ingestion.")
            return

        logger.info(f"Embedding {len(texts)} baseline papers...")
        vectors = self.embedder.embed(texts)
        for p, v in zip(papers, vectors):
            p.vector = v
        
        logger.info(f"Saving {len(papers)} papers to {self.store.table_name}...")
        self.store.save_papers(papers)
        logger.info("Baseline ingestion complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="category-agnostic novelty detection")
    args = parser.parse_args()
    
    detector = Mahalanbois76()
    detector.run(args.query)
