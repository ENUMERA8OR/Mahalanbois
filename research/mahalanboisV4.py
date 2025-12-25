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
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# UI & CLI
import streamlit as st
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import box

# Transformers for Local SLM
import ollama

# Setup Logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("mahalanbois_v4")

@dataclass
class Paper:
    title: str
    summary: str
    url: str
    published: str
    authors: List[str]
    novelty_score: float = 0.0
    cluster: int = -1
    explanation: str = ""
    vector: Optional[np.ndarray] = None

    def to_dict(self):
        d = asdict(self)
        if d['vector'] is not None:
            d['vector'] = d['vector'].tolist()
        return d

class LanceStore:
    """LanceDB wrapper for persistent scientific vector storage."""
    def __init__(self, db_path: str = ".mahalanbois_db"):
        self.db = lancedb.connect(db_path)
        self.table_name = "papers"

    def save_papers(self, papers: List[Paper]):
        data = [p.to_dict() for p in papers]
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data=data)

    def get_history(self, limit: int = 1000) -> List[Paper]:
        if self.table_name not in self.db.table_names():
            return []
        table = self.db.open_table(self.table_name)
        df = table.to_pandas()
        # Convert back to Paper objects
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

class Embedder:
    """Nomic-Embed-Text-v1.5 for long context scientific embeddings."""
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        logger.info(f"Loading Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        # Nomic requires instruction prefix
        prefix = "search_query: " if is_query else "search_document: "
        prefixed_texts = [prefix + t for t in texts]
        return self.model.encode(prefixed_texts, convert_to_numpy=True)

class NoveltyEngine:
    """The Mathematical Heart: GMM + Cluster-Conditional Mahalanobis."""
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        self.cluster_models = {} # Store LedoitWolf estimators per cluster

    def fit(self, X: np.ndarray):
        """Fits GMM clusters and local covariance models to historical data."""
        logger.info(f"Fitting GMM with {self.n_clusters} clusters...")
        self.gmm.fit(X)
        labels = self.gmm.predict(X)
        
        for i in range(self.n_clusters):
            X_cluster = X[labels == i]
            if len(X_cluster) > X.shape[1]: # Need enough samples for robust covariance
                lw = LedoitWolf()
                lw.fit(X_cluster)
                self.cluster_models[i] = lw
            else:
                self.cluster_models[i] = None # Fallback to global or euclidean

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates distance relative to the closest GMM cluster."""
        probs = self.gmm.predict_proba(X)
        closest_clusters = np.argmax(probs, axis=1)
        scores = np.zeros(len(X))
        
        for i, (vec, cluster_idx) in enumerate(zip(X, closest_clusters)):
            model = self.cluster_models.get(cluster_idx)
            if model:
                # Mahalanobis distance within the specific topic cluster
                scores[i] = model.mahalanobis(vec.reshape(1, -1))[0]
            else:
                # Fallback to simple distance from cluster mean
                mean = self.gmm.means_[cluster_idx]
                scores[i] = np.linalg.norm(vec - mean)
        
        return scores, closest_clusters

class ReasoningAgent:
    """The Judge: Qwen3-0.6B (Local SLM) for explanation via Ollama."""
    def __init__(self, model_name: str = "qwen3:0.6b"): 
        logger.info(f"Initializing Reasoning Agent with Ollama model: {model_name}...")
        self.model_name = model_name

    def explain_novelty(self, candidate: Paper, neighbors: List[Paper]) -> str:
        """Generates a brief explanation of why the paper is statistically novel."""
        context = "\n".join([f"- {n.title}: {n.summary[:200]}..." for n in neighbors])
        
        prompt = f"""[SCIENTIFIC COMPARISON TASK]
Compare the Candidate Paper against the Established Papers in the same field.
Established Papers (Historical Context):
{context}

Candidate Paper:
Title: {candidate.title}
Abstract: {candidate.summary}

Task: Is the Candidate Paper novel because of a new METHOD, a new DATASET, or a new APPLICATION? 
Explain in 2 sentences maximum. Output ONLY JSON format: {{"novelty_type": "...", "justification": "..."}}
"""
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            return "Explanation failed."

class MahalanboisV4:
    """Main Orchestrator for Research Intelligence."""
    def __init__(self):
        self.store = LanceStore()
        self.embedder = Embedder()
        self.engine = NoveltyEngine()
        self.judge = None # Lazy load only when needed

    def ingest_history(self, query: str = "cat:cs.AI OR cat:cs.LG", limit: int = 500):
        logger.info(f"Ingesting history for: {query}")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.SubmittedDate)
        
        papers = []
        summaries = []
        for res in client.results(search):
            p = Paper(
                title=res.title,
                summary=res.summary,
                url=res.entry_id,
                published=res.published.isoformat(),
                authors=[a.name for a in res.authors]
            )
            papers.append(p)
            summaries.append(res.summary)
        
        vectors = self.embedder.embed(summaries)
        for p, vec in zip(papers, vectors):
            p.vector = vec
            
        self.store.save_papers(papers)
        logger.info(f"Successfully ingested {len(papers)} papers into LanceDB.")

    def run_analysis(self, query: str, top_n: int = 5):
        # 1. Load History
        history = self.store.get_history(limit=2000)
        if not history:
            self.ingest_history()
            history = self.store.get_history()

        X_hist = np.array([p.vector for p in history])
        
        # 2. Fit Engine
        self.engine.fit(X_hist)
        
        # 3. Fetch Query Papers
        logger.info(f"Fetching new papers for: {query}...")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate)
        
        query_papers = []
        query_summaries = []
        for res in client.results(search):
            p = Paper(
                title=res.title,
                summary=res.summary,
                url=res.entry_id,
                published=res.published.isoformat(),
                authors=[a.name for a in res.authors]
            )
            query_papers.append(p)
            query_summaries.append(res.summary)
            
        X_query = self.embedder.embed(query_summaries, is_query=True)
        
        # 4. Score Novelty
        novelty_scores, clusters = self.engine.score(X_query)
        for i, p in enumerate(query_papers):
            p.novelty_score = novelty_scores[i]
            p.cluster = clusters[i]
            p.vector = X_query[i]

        # 5. Rank and Explain
        outliers = sorted(query_papers, key=lambda x: x.novelty_score, reverse=True)[:top_n]
        
        if not self.judge:
            self.judge = ReasoningAgent()
            
        for p in outliers:
            # Find neighbors in history for contrastive CoT
            cluster_hist = [h for h in history if h.cluster == p.cluster]
            # Simple fallback to all history if cluster empty
            neighbors = cluster_hist[:5] if cluster_hist else history[:5]
            p.explanation = self.judge.explain_novelty(p, neighbors)

        # 6. Display Results
        table = Table(title="Mahalanbois V4: Weak Signal Detection", box=box.DOUBLE)
        table.add_column("Score", style="magenta")
        table.add_column("Topic", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Explanation (SLM)", style="dim")

        for p in outliers:
            table.add_row(
                f"{p.novelty_score:.2f}",
                f"Cluster {p.cluster}",
                p.title,
                p.explanation
            )
        
        console.print(table)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="cat:cs.AI")
    parser.add_argument("--ingest", action="store_true")
    args = parser.parse_args()
    
    agent = MahalanboisV4()
    if args.ingest:
        agent.ingest_history()
    
    agent.run_analysis(args.query)
