import os
import json
import argparse
import logging
import numpy as np
import arxiv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
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
import plotly.express as px

# Setup Rich Console
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mahalanbois-v3.5")

@dataclass
class Document:
    title: str
    summary: str
    url: str
    date: str
    authors: List[str]
    embedding: Optional[np.ndarray] = None
    score_local_md: float = 0.0
    score_lid: float = 0.0
    is_reference: bool = False

class ArxivSource:
    """Fetches data from ArXiv with caching."""
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

        with open(cache_file, 'w', encoding='utf-8') as f:
            data = [{k: v for k, v in d.__dict__.items() if k not in ['embedding', 'score_local_md', 'score_lid', 'is_reference']} for d in docs]
            json.dump(data, f, indent=2)
        
        return docs

class AdvancedNoveltyEngine:
    """The Math Core V3.5: Local Mahalanobis + Local Intrinsic Dimensionality (LID)."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def compute_local_mahalanobis(self, X_query: np.ndarray, X_ref: np.ndarray, k: int = 40) -> np.ndarray:
        """
        Computes Mahalanobis distance relative to the k-nearest historical neighbors.
        This handles multi-modal distributions (different research clusters).
        """
        logger.info(f"Computing Local Mahalanobis (k={k})...")
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X_ref)
        distances, indices = nbrs.kneighbors(X_query)
        
        scores = []
        for i, q_point in enumerate(X_query):
            # Local cluster from history
            local_ref = X_ref[indices[i]]
            
            # Robust local covariance
            try:
                lw = LedoitWolf(assume_centered=False)
                lw.fit(local_ref)
                # cov_inv = lw.precision_ (Mahalanobis distance tool usually wants the precison)
                # But sklearn's mahalanobis method is easier if we just use the estimator
                dist = lw.mahalanobis(q_point.reshape(1, -1))[0]
                scores.append(dist)
            except Exception:
                # Fallback to local euclidean
                scores.append(np.mean(distances[i]))
        
        return np.array(scores)

    def compute_lid(self, X_candidates: np.ndarray, X_ref: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Computes Local Intrinsic Dimensionality (LID).
        Identifies papers that sit on 'dimensionally discordant' manifolds.
        """
        logger.info(f"Computing Local Intrinsic Dimensionality (k={k})...")
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref)
        distances, _ = nbrs.kneighbors(X_candidates)
        
        # LID formula: - (1/k * sum(log(r_i / r_max)))^-1
        # Add epsilon to avoid log(0)
        eps = 1e-10
        r_max = distances[:, -1]
        
        # Calculate log ratios
        # distances[:, :-1] / r_max[:, None]
        lids = []
        for i in range(len(X_candidates)):
            r_i = distances[i]
            # Avoid division by zero if r_max is 0
            if r_max[i] < eps:
                lids.append(0.0)
                continue
                
            # Filter out zero distances for the log
            valid_r = r_i[r_i > eps]
            if len(valid_r) < 2:
                lids.append(0.0)
                continue
                
            log_sum = np.sum(np.log(valid_r / r_max[i]))
            lid = - (len(valid_r) / log_sum) if log_sum != 0 else 0.0
            lids.append(lid)
            
        return np.array(lids)

class Visualizer:
    @staticmethod
    def generate_map(docs: List[Document], filename="mahalanobis_v35_map.html"):
        """Generates an interactive 2D map of the paper landscape."""
        if not docs or docs[0].embedding is None:
            return

        embeddings = np.array([d.embedding for d in docs])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)

        titles = [d.title for d in docs]
        dates = [d.date.split("T")[0] for d in docs]
        types = ["Reference (History)" if d.is_reference else "Query (New)" for d in docs]
        
        # Use LID as the intensity for new papers
        intensities = [d.score_lid if not d.is_reference else 0 for d in docs]
        
        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=types,
            size=[12 if i > 0 else 5 for i in intensities],
            hover_name=titles,
            hover_data={"Date": dates, "LID Score": [f"{i:.2f}" for i in intensities]},
            title="Mahalanbois V3.5: Local Subspace & Dimensionality Outliers",
            color_discrete_map={"Reference (History)": "#333333", "Query (New)": "#ff4b4b"},
            template="plotly_dark"
        )

        fig.write_html(filename)
        logger.info(f"Visualization saved to {filename}")

class AnalystV35:
    def __init__(self):
        self.source = ArxivSource()
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.engine = AdvancedNoveltyEngine()

    def run(self, query: str, total_limit: int = 300, query_size: int = 50):
        console.print(Panel(
            "[bold cyan]MAHALANBOIS V3.5[/bold cyan]\n[bold white]LOCAL SUBSPACE & INTRINSIC DIMENSIONALITY[/bold white]",
            title="[bold green]Weak Signal Detection[/bold green]",
            subtitle=f"[dim]Metrics: Local Mahalanobis + LID | Query: {query}[/dim]"
        ))

        # 1. Fetch
        with console.status("[bold green]Harvesting Intelligence..."):
            all_docs = self.source.fetch(query, total_limit)
        
        if not all_docs:
            return

        # 2. Time Split
        query_docs = all_docs[:query_size]
        reference_docs = all_docs[query_size:]
        for d in reference_docs: d.is_reference = True

        # 3. Embed
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True) as progress:
            task = progress.add_task("[cyan]Embedding Documents...", total=len(all_docs))
            texts = [d.summary for d in all_docs]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            for i, doc in enumerate(all_docs):
                doc.embedding = embeddings[i]
                progress.advance(task)

        X_ref = np.array([d.embedding for d in reference_docs])
        X_query = np.array([d.embedding for d in query_docs])

        # 4. Local Mahalanobis (Subspace Outlier Detection)
        with console.status("[bold blue]Calculating Local Mahalanobis Outliers..."):
            md_scores = self.engine.compute_local_mahalanobis(X_query, X_ref)
            for i, score in enumerate(md_scores):
                query_docs[i].score_local_md = score

        # 5. Filter & LID Reranking
        # Take the top Mahalanobis outliers to check their Intrinsic Dimensionality
        candidate_count = int(query_size * 0.4)
        top_md_indices = np.argsort(md_scores)[-candidate_count:]
        X_candidates = X_query[top_md_indices]
        
        with console.status("[bold magenta]Analyzing Intrinsic Dimensionality (LID)..."):
            lid_scores = self.engine.compute_lid(X_candidates, X_ref)
            for i, idx in enumerate(top_md_indices):
                query_docs[idx].score_lid = lid_scores[i]

        # 6. Final Sort (By LID)
        results = sorted([d for d in query_docs if d.score_lid > 0], key=lambda x: x.score_lid, reverse=True)

        # 7. Report
        table = Table(title="💎 V3.5 DIMENSIONAL OUTLIER REPORT 💎", box=box.HORIZONTALS)
        table.add_column("Rank", style="cyan")
        table.add_column("LID", style="bold magenta")
        table.add_column("Loc-MD", style="dim")
        table.add_column("Title", style="white")

        for i, doc in enumerate(results[:5]):
            table.add_row(
                str(i+1),
                f"{doc.score_lid:.2f}",
                f"{doc.score_local_md:.2f}",
                doc.title
            )
        
        console.print(table)
        
        # 8. Visualize
        Visualizer.generate_map(all_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="cat:cs.AI OR cat:cs.LG")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    AnalystV35().run(args.query, args.total, args.batch)
