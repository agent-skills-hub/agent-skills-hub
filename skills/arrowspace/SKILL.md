---
name: arrowspace
description: "Spectral vector search using graph Laplacian eigenstructure. Use when cosine/L2 similarity misses latent structure, or you need λτ-indexed retrieval with spectral awareness."
risk: safe
source: self
---

# ArrowSpace

Spectral vector search that augments nearest-neighbour search with graph Laplacian features. Computes a Laplacian over the item graph and uses the Rayleigh quotient to produce a λτ (lambda-tau) score per item, enabling search that respects both semantic similarity and structural role.

## Use this skill when

- Cosine or L2 similarity misses latent structure in your embeddings
- You want graph-based retrieval with spectral awareness
- You need to characterise the spectral properties of an embedding space
- You are building RAG pipelines where contextual role matters alongside semantic content

## Do not use this skill when

- Pure cosine or L2 similarity is sufficient
- Your dataset has fewer than 10 items (graph structure is not meaningful)
- You need real-time indexing of streaming data (ArrowSpace is batch-oriented)

## Installation

```bash
pip install arrowspace
```

## Quick Start

```python
from arrowspace import ArrowSpaceBuilder
import numpy as np

items = np.array([[0.1, 0.2, 0.3],
                  [0.0, 0.5, 0.1],
                  [0.9, 0.1, 0.0]], dtype=np.float64)

graph_params = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
builder = ArrowSpaceBuilder(items, graph_params=graph_params)
aspace = builder.build()
lambdas = aspace.lambdas()
```

## Instructions

1. Install `arrowspace` via pip.
2. Prepare your embedding vectors as an (N, d) NumPy array of `float64`.
3. Configure `graph_params`: `eps` (neighbourhood radius), `k` (graph degree), `p` (Minkowski norm), `sigma` (Gaussian kernel width), `topk` (retrieval count).
4. Call `ArrowSpaceBuilder(items, graph_params).build()` to construct the spectral space.
5. Query with `aspace.lambdas()` (array indexed by insertion order) or `aspace.lambdas_sorted()` (sorted by score ascending).
6. Higher λτ values indicate items that are both semantically close to the query and structurally central in the graph.

## Best Practices

- Normalise embeddings to unit norm before passing to ArrowSpace (the builder does this internally, but pre-normalising helps with graph construction).
- Start with `eps` proportional to `1 / sqrt(dim)` and tune from there.
- Use `k` between 3 and 25 depending on dataset size (rule of thumb: N / 50).
- Set `sigma = None` to auto-select kernel width based on distance distribution.

## Related Skills

- `vector-database-engineer` — General vector database expertise
- `embedding-strategies` — Embedding model selection and chunking
- `similarity-search-patterns` — Semantic search implementation patterns
- `hybrid-search-implementation` — Combined semantic + keyword search
