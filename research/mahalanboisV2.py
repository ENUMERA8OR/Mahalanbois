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
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class PaperFetcher:
    """Handles fetching and caching of data from ArXiv."""
    def __init__(self, cache_dir: str = ".arxiv_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch(self, query: str, max_results: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        safe_query = "".join(c if c.isalnum() else "_" for c in query)
        cache_file = os.path.join(self.cache_dir, f"{safe_query}_{max_results}.json")

        if not force_refresh and os.path.exists(cache_file):
            logger.info(f"Loading {max_results} papers from cache: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted. Re-fetching.")

        logger.info(f"Fetching {max_results} papers from arXiv for query: '{query}'...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers_data = []
        try:
            for result in client.results(search):
                papers_data.append({
                    "title": result.title,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "date": result.published.isoformat(),
                    "authors": [a.name for a in result.authors]
                })
        except Exception as e:
            logger.error(f"Failed to fetch papers: {e}")
            return []

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2)
        
        logger.info(f"Successfully fetched and cached {len(papers_data)} papers.")
        return papers_data

class Embedder:
    """Handles text embedding generation."""
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        
        logger.info(f"Encoding {len(texts)} texts...")
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

class NoveltyDetector:
    """Core logic for Kernel Mahalanobis + MA-DPR detection."""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _estimate_gamma(self, X: np.ndarray) -> float:
        """Heuristic to estimate RBF kernel gamma based on median pairwise distance."""
        # Sample subset if too large
        if X.shape[0] > 1000:
            idx = self.rng.choice(X.shape[0], 1000, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        dists = pdist(X_sample, metric='sqeuclidean')
        median_dist = np.median(dists)
        if median_dist == 0:
            return 1.0
        # gamma = 1 / (2 * sigma^2). Often set to 1/median_dist for heuristic.
        gamma = 1.0 / median_dist
        logger.info(f"Estimated heuristic gamma: {gamma:.4f}")
        return gamma

    def compute_kernel_mahalanobis(self, X: np.ndarray, n_components: int = 512) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Computing Robust Kernel Mahalanobis novelty...")
        d_in = X.shape[1]
        
        gamma = self._estimate_gamma(X)
        
        # RFF Mapping
        W = self.rng.normal(scale=np.sqrt(2 * gamma), size=(d_in, n_components))
        b = self.rng.uniform(0, 2 * np.pi, size=n_components)
        Z = np.sqrt(2 / n_components) * np.cos(X @ W + b)
        
        # Robust Covariance (Ledoit-Wolf)
        try:
            cov = LedoitWolf(store_precision=True, assume_centered=False)
            cov.fit(Z)
            mahal_dists = cov.mahalanobis(Z)
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed ({e}), using Diagonal fallback.")
            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z)
            mahal_dists = np.linalg.norm(Z_scaled, axis=1)

        return mahal_dists, Z

    def compute_madpr(self, Z_candidates: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
        logger.info(f"Running MA-DPR on {len(Z_candidates)} candidates...")
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(Z_candidates)-1), metric="euclidean").fit(Z_candidates)
        adj_matrix = nbrs.kneighbors_graph(Z_candidates, mode="distance")
        
        dist_matrix = shortest_path(adj_matrix, method='auto', directed=True, return_predecessors=False)
        
        # Handle disconnected components
        finite_dists = dist_matrix[np.isfinite(dist_matrix)]
        if len(finite_dists) > 0:
            penalty = finite_dists.max() * 2.0
            dist_matrix[~np.isfinite(dist_matrix)] = penalty
        else:
            dist_matrix[:] = 0

        return np.mean(dist_matrix, axis=1)

class MahalanboisPipeline:
    def __init__(self, query: str, max_results: int = 200):
        self.query = query
        self.max_results = max_results
        self.fetcher = PaperFetcher()
        # Upgraded model for better semantic resolution
        self.embedder = Embedder(model_name="all-mpnet-base-v2")
        self.detector = NoveltyDetector()

    def run(self, top_k_candidates: int = 30, final_top_k: int = 5):
        # 1. Fetch
        papers = self.fetcher.fetch(self.query, self.max_results)
        if not papers:
            return

        # 2. Embed
        texts = [p['summary'] for p in papers]
        X = self.embedder.embed(texts)

        # 3. Global Filter (Kernel Mahalanobis)
        km_scores, Z = self.detector.compute_kernel_mahalanobis(X)
        
        # Select candidates
        candidate_indices = np.argsort(km_scores)[-top_k_candidates:]
        
        # 4. Local Filter (MA-DPR)
        Z_candidates = Z[candidate_indices]
        madpr_scores = self.detector.compute_madpr(Z_candidates)
        
        # 5. Final Ranking
        final_local_rank = np.argsort(madpr_scores)[-final_top_k:]
        final_global_indices = candidate_indices[final_local_rank][::-1]

        # 6. Report
        print(f"\n🔥 Top {final_top_k} Weak-Signal Papers (Mahalanobis + MA-DPR):\n")
        print("="*80)
        for idx in final_global_indices:
            p = papers[idx]
            print(f"• {p['title']}")
            print(f"  Authors: {', '.join(p['authors'][:3])}")
            print(f"  Link: {p['url']}")
            print(f"  Summary: {p['summary'][:200].replace(chr(10), ' ')}...")
            print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mahalanbois: See Sooner, Act Faster.")
    parser.add_argument("--query", type=str, default="cat:cs.AI OR cat:cs.LG", help="ArXiv search query")
    parser.add_argument("--max_results", type=int, default=200, help="Number of papers to fetch")
    
    args = parser.parse_args()

    pipeline = MahalanboisPipeline(args.query, args.max_results)
    pipeline.run()