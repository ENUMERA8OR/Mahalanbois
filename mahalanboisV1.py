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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.covariance import LedoitWolf
from scipy.sparse.csgraph import shortest_path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

class ArxivWeakSignalDetector:
    def __init__(self, 
                 query: str = "cat:cs.AI OR cat:cs.LG", 
                 max_results: int = 200, 
                 cache_dir: str = ".arxiv_cache",
                 model_name: str = "all-MiniLM-L6-v2",
                 seed: int = 42):
        self.query = query
        self.max_results = max_results
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def fetch_papers(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetches papers from arXiv with caching to avoid redundant API calls.
        """
        # Create a safe filename from query
        safe_query = "".join(c if c.isalnum() else "_" for c in self.query)
        cache_file = os.path.join(self.cache_dir, f"{safe_query}_{self.max_results}.json")

        if not force_refresh and os.path.exists(cache_file):
            logger.info(f"Loading {self.max_results} papers from cache: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert dict back to objects if necessary, or just return dicts
                    # For this script, dicts are easier to handle
                    return data
            except json.JSONDecodeError:
                logger.warning("Cache file corrupted. Re-fetching.")

        logger.info(f"Fetching {self.max_results} papers from arXiv for query: '{self.query}'...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers_data = []
        try:
            # client.results returns a generator
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
            if not papers_data:
                return []

        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2)
        
        logger.info(f"Successfully fetched and cached {len(papers_data)} papers.")
        return papers_data

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized embeddings for the provided texts.
        """
        logger.info(f"Encoding {len(texts)} abstracts using {self.model_name}...")
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings

    from sklearn.covariance import LedoitWolf

    def compute_rff_novelty(self, X: np.ndarray, n_components: int = 512, gamma: float = 1.0) -> np.ndarray:
        """
        Computes Kernel Mahalanobis novelty using Random Fourier Features (RFF)
        and the Ledoit-Wolf robust covariance estimator.
        """
        logger.info("Computing Kernel Mahalanobis novelty (Distance to Centroid)...")
        d_in = X.shape[1]
        
        # RFF Mapping
        # We use a fixed seed for the random projection to ensure stability across runs if needed
        # (Already handled by self.rng)
        W = self.rng.normal(scale=np.sqrt(2 * gamma), size=(d_in, n_components))
        b = self.rng.uniform(0, 2 * np.pi, size=n_components)
        
        # Z is the explicit feature map of the RBF kernel
        Z = np.sqrt(2 / n_components) * np.cos(X @ W + b)
        
        # Use Ledoit-Wolf shrinkage to estimate covariance in high dimensions (d > n)
        # This provides a robust Mahalanobis distance even with few samples
        try:
            cov_estimator = LedoitWolf(store_precision=True, assume_centered=False)
            cov_estimator.fit(Z)
            
            # Mahalanobis distance of each point to the distribution of Z
            # dist = sqrt((z - mu)^T S^-1 (z - mu))
            mahalanobis_dists = cov_estimator.mahalanobis(Z)
            
            # Use distances as novelty scores (higher = further from distribution center)
            novelty_scores = mahalanobis_dists
            
        except Exception as e:
            logger.warning(f"Ledoit-Wolf estimation failed ({e}), falling back to Diagonal Mahalanobis (StandardScaler).")
            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z)
            # Distance to mean (0) in standardized space is just the norm
            novelty_scores = np.linalg.norm(Z_scaled, axis=1)

        return novelty_scores, Z

    def compute_madpr_rerank(self, Z_candidates: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
        """
        Reranks candidates using Max-Avg Dissimilarity via Path Reasoning (MA-DPR) logic.
        Uses shortest path on the KNN graph.
        """
        logger.info(f"Reranking {len(Z_candidates)} candidates using Graph Path analysis...")
        
        # Build KNN Graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(Z_candidates)
        # mode='distance' returns actual euclidean distances as weights
        adj_matrix = nbrs.kneighbors_graph(Z_candidates, mode="distance")
        
        # Compute all-pairs shortest paths efficiently using Floyd-Warshall or Dijkstra
        # directed=False ensures symmetric graph flow if needed, but KNN is inherently directed.
        # However, for manifold traversal, treating it as undirected usually stabilizes small graphs.
        # We'll keep directed=True to respect the K-nearest relationship strictness.
        dist_matrix = shortest_path(adj_matrix, method='auto', directed=True, return_predecessors=False)
        
        # Handle disconnected components (infinite distances)
        # Replace inf with a penalty (e.g., 2x the max finite distance) to avoid NaN or Inf
        finite_dists = dist_matrix[np.isfinite(dist_matrix)]
        if len(finite_dists) > 0:
            max_dist = finite_dists.max()
            penalty = max_dist * 2.0
            dist_matrix[~np.isfinite(dist_matrix)] = penalty
        else:
            # Fallback if graph is totally disconnected (rare)
            dist_matrix[:] = 0

        # Score is average shortest path to all other nodes in the candidate set
        madpr_scores = np.mean(dist_matrix, axis=1)
        
        return madpr_scores

    def run(self, top_k_candidates: int = 30, final_top_k: int = 5):
        # 1. Fetch
        papers = self.fetch_papers()
        if not papers:
            logger.error("No papers found. Exiting.")
            return

        texts = [p['summary'] for p in papers]

        # 2. Embed
        X = self.get_embeddings(texts)

        # 3. Kernel Mahalanobis (Global Novelty)
        km_scores, Z = self.compute_rff_novelty(X)

        # 4. Shortlist Top-K Candidates
        # argsort returns indices of sorted elements (ascending), so we take the last K
        candidate_indices = np.argsort(km_scores)[-top_k_candidates:]
        
        # 5. Local Manifold Reranking (MA-DPR)
        # We only look at the manifold structure OF THE CANDIDATES (or X[candidates]?)
        # The original script projected Z[candidates]. Let's stick to that.
        Z_candidates = Z[candidate_indices]
        
        madpr_scores = self.compute_madpr_rerank(Z_candidates)
        
        # 6. Final Ranking
        # Sort candidates by their MA-DPR score
        final_rank_local_indices = np.argsort(madpr_scores)[-final_top_k:]
        final_global_indices = candidate_indices[final_rank_local_indices]
        
        # Reverse to show highest score first
        final_global_indices = final_global_indices[::-1]

        print(f"\n🔥 Top {final_top_k} Weak-Signal Papers (Kernel MD → MA-DPR):\n")
        print("="*80)
        for idx in final_global_indices:
            p = papers[idx]
            print(f"• {p['title']}")
            print(f"  Authors: {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}")
            print(f"  Link: {p['url']}")
            print(f"  Summary: {p['summary'][:200].replace(chr(10), ' ')}...")
            print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find novel arXiv papers using Kernel Mahalanobis & Graph Reranking.")
    parser.add_argument("--query", type=str, default="cat:cs.AI OR cat:cs.LG", help="ArXiv search query")
    parser.add_argument("--max_results", type=int, default=200, help="Number of papers to fetch")
    parser.add_argument("--candidates", type=int, default=30, help="Number of candidates for reranking")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final papers to display")
    parser.add_argument("--refresh", action="store_true", help="Force refresh of cache")
    
    args = parser.parse_args()

    detector = ArxivWeakSignalDetector(
        query=args.query,
        max_results=args.max_results
    )
    
    detector.run(
        top_k_candidates=args.candidates,
        final_top_k=args.top_k
    )
