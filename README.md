# Mini RL-Index

A lightweight implementation of the RL-Index idea from the paper:

**RL-Index: Reinforcement Learning for Retrieval Index Reasoning**

This project demonstrates how retrieval quality can be improved by generating reasoning-based retrieval rationales for documents before indexing them.

Instead of making an LLM reason at query time, we enrich documents during indexing with explanations describing:

* Why a document may be retrieved
* Related concepts
* Alternative phrasings
* Potential user intents

The enriched documents are then embedded and stored in a FAISS vector index.

---

## Motivation

Traditional RAG systems often perform query-side reasoning:

```text
User Query
    ↓
LLM Query Rewrite
    ↓
Retriever
    ↓
Documents
```

This requires additional LLM calls for every search.

Mini RL-Index shifts reasoning to indexing time:

```text
Documents
    ↓
Generate Rationales
    ↓
Augment Documents
    ↓
Embeddings
    ↓
FAISS Index
```

This allows retrieval to benefit from reasoning without incurring query-time reasoning costs.

---

## Architecture

### Offline Index Construction

```text
Document
    ↓
OpenRouter LLM
    ↓
Generate Multiple Rationales
    ↓
Reward Evaluation
    ↓
Select Best Rationale
    ↓
Augmented Document
    ↓
BGE Embedding
    ↓
FAISS Index
```

### Online Retrieval

```text
User Query
    ↓
BGE Embedding
    ↓
FAISS Search
    ↓
Top Documents
```

---

## Features

* OpenRouter-based rationale generation
* Multiple rationale candidates per document
* Reward-based rationale selection
* BGE embeddings
* FAISS vector search
* Colab T4 compatible
* No model fine-tuning required

---

## Installation

```bash
pip install sentence-transformers
pip install faiss-cpu
pip install openai
pip install numpy
pip install pandas
```

---

## Models Used

### Rationale Generation

Any OpenRouter-compatible model:

* deepseek/deepseek-chat-v3
* qwen/qwen-3-32b
* mistralai/mistral-small
* moonshotai/kimi-k2

### Embeddings

```python
BAAI/bge-small-en-v1.5
```

---

## OpenRouter Setup

```python
from openai import OpenAI

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
```

Set your API key:

```python
OPENROUTER_API_KEY = "YOUR_KEY"
```

---

## Workflow

### Step 1: Create Documents

```python
documents = [
    "...",
    "...",
    "..."
]
```

### Step 2: Generate Rationales

Example rationale:

```text
Useful for:
- triangle congruence
- geometry proofs
- SAS theorem

Related concepts:
- equal triangles
- proof strategies
```

### Step 3: Generate Multiple Candidates

```text
Rationale A
Rationale B
Rationale C
Rationale D
```

### Step 4: Evaluate Reward

Current reward:

```python
reward = (
    similarity(query, augmented_doc)
    - similarity(query, original_doc)
)
```

Positive reward indicates the rationale improved retrieval alignment.

### Step 5: Select Best Rationale

```python
best_rationale = max(
    candidates,
    key=lambda r: reward(r)
)
```

### Step 6: Build Final Indexed Document

```text
DOCUMENT
+
BEST RETRIEVAL RATIONALE
```

### Step 7: Create Embeddings

```python
vectors = embedder.encode(
    final_docs,
    normalize_embeddings=True
)
```

### Step 8: Build FAISS Index

```python
index = faiss.IndexFlatIP(dimension)
index.add(vectors)
```

### Step 9: Retrieve

```python
scores, ids = index.search(
    query_embedding,
    k=3
)
```

---

## Example

### Original Document

```text
Binary Search Tree supports ordered lookup.
```

### Generated Rationale

```text
Useful when searching for:

- fast lookup
- efficient search
- ordered data structures
- logarithmic search
```

### Indexed Version

```text
Binary Search Tree supports ordered lookup.

RETRIEVAL RATIONALE:

Useful when searching for:
- fast lookup
- efficient search
- ordered data structures
- logarithmic search
```

---

## Comparison

### Baseline Retrieval

```text
Document
    ↓
Embedding
    ↓
FAISS
```

### Mini RL-Index

```text
Document
    ↓
Generate Rationale
    ↓
Document + Rationale
    ↓
Embedding
    ↓
FAISS
```

---

## Current Limitations

This project is not a full reproduction of RL-Index.

Differences:

* No GRPO training
* No policy optimization
* No retriever-in-the-loop RL
* Reward is heuristic
* Small-scale evaluation

The goal is to demonstrate the core intuition behind RL-Index using a lightweight implementation.

---

## Future Improvements

### True RL Training

Train a small model using:

* TRL
* GRPO
* Qwen 0.5B

### Better Reward Functions

Use:

* Recall@K
* MRR
* NDCG
* Retriever score improvements

### Larger Corpora

Test on:

* Wikipedia
* SQuAD
* AG News
* Domain-specific datasets

### Production RAG Integration

Integrate with:

* LangChain
* LlamaIndex
* Haystack
* LightRAG

---

## Results to Measure

Evaluate:

* Recall@1
* Recall@5
* Hit Rate
* MRR

Compare:

```text
Baseline Index
vs
Mini RL-Index
```

to determine whether retrieval rationales improve search quality.

---

## Acknowledgements

Inspired by:

RL-Index: Reinforcement Learning for Retrieval Index Reasoning

This repository provides a lightweight educational implementation of the paper's core idea for experimentation on Google Colab.
