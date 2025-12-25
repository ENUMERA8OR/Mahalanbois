import os
import json
import time
import argparse
import logging
import numpy as np
import arxiv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Rich for CLI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich import box

# Plotly for Visualization
import plotly.graph_objects as go
import plotly.express as px

# Setup Rich Console
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mahalanbois")

@dataclass
class Document:
    title: str
    summary: str
    url: str
    date: str
    authors: List[str]
    embedding: Optional[np.ndarray] = None
    score_global: float = 0.0
    score_local: float = 0.0
    is_reference: bool = False

class ArxivSource:
    """Fetches data from ArXiv."""
    def __init__(self, cache_dir: str = ".arxiv_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch(self, query: str, max_results: int, force_refresh: bool = False) -> List[Document]:
        safe_query = "".join(c if c.isalnum() else "_" for c in query)
        cache_file = os.path.join(self.cache_dir, f"{safe_query}_{max_results}.json")

        if not force_refresh and os.path.exists(cache_file):
            logger.info(f"Loading {max_results} papers from cache...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Document(**d) for d in data]

        logger.info(f"Fetching {max_results} papers from arXiv...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        docs = []
        try:
            for result in client.results(search):
                doc = Document(
                    title=result.title,
                    summary=result.summary,
                    url=result.entry_id,
                    date=result.published.isoformat(),
                    authors=[a.name for a in result.authors]
                )
                docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to fetch: {e}")
            return []

        # Serialize for cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            # exclude embedding/scores from cache for now to keep it clean JSON
            data = [{k: v for k, v in d.__dict__.items() if k not in ['embedding', 'score_global', 'score_local', 'is_reference']} for d in docs]
            json.dump(data, f, indent=2)
        
        return docs

class NoveltyEngine:
    """The Math Core: RFF + Ledoit-Wolf + MA-DPR."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.cov_estimator = None
        self.W = None
        self.b = None
        self.rff_dim = 512
        self.gamma = 1.0

    def fit_reference(self, X_ref: np.ndarray):
        """Trains the normality model on the Reference (History) set."""
        d_in = X_ref.shape[1]
        
        # 1. Heuristic Gamma
        dists = pdist(X_ref[:1000], metric='sqeuclidean') # Subsample if large
        median_dist = np.median(dists)
        self.gamma = 1.0 / (median_dist if median_dist > 0 else 1.0)
        logger.info(f"Reference Gamma: {self.gamma:.4f}")

        # 2. RFF Mapping Setup
        self.W = self.rng.normal(scale=np.sqrt(2 * self.gamma), size=(d_in, self.rff_dim))
        self.b = self.rng.uniform(0, 2 * np.pi, size=self.rff_dim)

        # 3. Project Reference
        Z_ref = np.sqrt(2 / self.rff_dim) * np.cos(X_ref @ self.W + self.b)

        # 4. Robust Covariance (Ledoit-Wolf)
        try:
            self.cov_estimator = LedoitWolf(store_precision=True, assume_centered=False)
            self.cov_estimator.fit(Z_ref)
        except Exception as e:
            logger.warning(f"Ledoit-Wolf fit failed: {e}")
            self.cov_estimator = None # Will fallback to euclidean later if None

    def score_novelty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scores X against the trained reference model."""
        if self.W is None:
            raise ValueError("Engine not fitted. Call fit_reference() first.")

        # Project
        Z = np.sqrt(2 / self.rff_dim) * np.cos(X @ self.W + self.b)

        # Mahalanobis Distance
        if self.cov_estimator:
            scores = self.cov_estimator.mahalanobis(Z)
        else:
            # Fallback
            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z)
            scores = np.linalg.norm(Z_scaled, axis=1)
        
        return scores, Z

    def compute_madpr(self, Z_candidates: np.ndarray, k: int = 8) -> np.ndarray:
        """Local Manifold Reranking."""
        if len(Z_candidates) < 2:
            return np.zeros(len(Z_candidates))
            
        nbrs = NearestNeighbors(n_neighbors=min(k, len(Z_candidates)-1), metric="euclidean").fit(Z_candidates)
        adj = nbrs.kneighbors_graph(Z_candidates, mode="distance")
        dist_matrix = shortest_path(adj, method='auto', directed=True, return_predecessors=False)
        
        finite_dists = dist_matrix[np.isfinite(dist_matrix)]
        if len(finite_dists) > 0:
            penalty = finite_dists.max() * 2.0
            dist_matrix[~np.isfinite(dist_matrix)] = penalty
        else:
            dist_matrix[:] = 0

        return np.mean(dist_matrix, axis=1)

class Visualizer:
    @staticmethod
    def generate_map(docs: List[Document], filename="mahalanobis_map.html"):
        """Generates an interactive 2D map of the paper landscape."""
        if not docs or docs[0].embedding is None:
            return

        embeddings = np.array([d.embedding for d in docs])
        
        # PCA Projection to 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)

        # Prepare Data for Plotly
        titles = [d.title for d in docs]
        dates = [d.date.split("T")[0] for d in docs]
        types = ["Reference (History)" if d.is_reference else "Query (New)" for d in docs]
        scores = [d.score_local if not d.is_reference else 0 for d in docs]
        sizes = [10 if not d.is_reference and d.score_local > 0 else 5 for d in docs] # Highlight anomalies

        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=types,
            size=sizes,
            hover_name=titles,
            hover_data={"Date": dates, "Novelty Score": [f"{s:.2f}" for s in scores]},
            title="Mahalanobis Landscape: Reference vs. Weak Signals",
            color_discrete_map={"Reference (History)": "lightgray", "Query (New)": "red"}
        )

        fig.update_layout(template="plotly_dark", showlegend=True)
        fig.write_html(filename)
        logger.info(f"Visualization saved to {filename}")

class Analyst:
    def __init__(self):
        self.source = ArxivSource()
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.engine = NoveltyEngine()

    def run(self, query: str, total_limit: int = 300, query_size: int = 50):
        console.print(Panel(
            "[bold magenta]MAHALANBOIS:[/bold magenta]\n[bold white]SEE SOONER, ACT FASTER[/bold white]",
            title="[bold green]Mahalanobis Pro[/bold green]",
            subtitle=f"[dim]Query: {query} | History: {total_limit-query_size} | Batch: {query_size}[/dim]"
        ))

        # 1. Fetch
        with console.status("[bold green]Fetching Intelligence..."):
            all_docs = self.source.fetch(query, total_limit)
        
        if not all_docs:
            console.print("[red]No docs found.[/red]")
            return

        # 2. Split Time (Newest = Query, Older = Reference)
        # ArXiv returns Newest first. 
        query_docs = all_docs[:query_size]      # The newest 50
        reference_docs = all_docs[query_size:]  # The older 250
        
        for d in reference_docs: d.is_reference = True
        for d in query_docs: d.is_reference = False

        if len(reference_docs) < 50:
            console.print("[yellow]Warning: Reference set too small for robust statistics.[/yellow]")

        # 3. Embed
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("[cyan]Embedding Documents...", total=len(all_docs))
            # Batch embedding
            texts = [d.summary for d in all_docs]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            for i, doc in enumerate(all_docs):
                doc.embedding = embeddings[i]
                progress.advance(task)

        X_ref = np.array([d.embedding for d in reference_docs])
        X_query = np.array([d.embedding for d in query_docs])

        # 4. Train Reference Model (Global Normality)
        with console.status("[bold blue]Fitting Historical Baseline (Ledoit-Wolf)..."):
            self.engine.fit_reference(X_ref)

        # 5. Score Query (Global Novelty)
        global_scores, Z_query = self.engine.score_novelty(X_query)
        for i, score in enumerate(global_scores):
            query_docs[i].score_global = score

        # 6. Filter & Rerank (Local MA-DPR)
        # We take top 50% of global outliers to rerank locally
        top_k_indices = np.argsort(global_scores)[-int(len(query_docs)*0.6):] 
        Z_candidates = Z_query[top_k_indices]
        
        madpr_scores = self.engine.compute_madpr(Z_candidates)
        
        # Assign local scores
        for i, idx in enumerate(top_k_indices):
            query_docs[idx].score_local = madpr_scores[i]

        # 7. Final Sort
        # Sort only by local score (which implies they passed the global filter)
        results = sorted([d for d in query_docs if d.score_local > 0], key=lambda x: x.score_local, reverse=True)

        # 8. Report
        table = Table(title="🔥 WEAK SIGNAL REPORT 🔥", box=box.ROUNDED)
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta")
        table.add_column("Title", style="white")
        table.add_column("Summary Snippet", style="dim")

        for i, doc in enumerate(results[:5]):
            table.add_row(
                str(i+1),
                f"{doc.score_local:.2f}",
                doc.title,
                doc.summary[:100].replace("\n", " ") + "..."
            )
        
        console.print(table)
        
        # 9. Visualize
        Visualizer.generate_map(all_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="cat:cs.AI OR cat:cs.LG")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    Analyst().run(args.query, args.total, args.batch)
