# HW2: Retrieval-Augmented Generation (RAG)

Implementation of an advanced RAG pipeline utilizing LightRAG and knowledge graph-based retrieval.

## Project Structure
- **hw2_baseline_code/**: Main implementation of the RAG system, incorporating the LightRAG framework.
- **database/**: Local document storage and indexed vector/graph data.
- **src/**: Support scripts for embedding generation and query processing.
- **lora/**: Adapter weights for model alignment with retrieval tasks.
- **fig/**: Visualizations of retrieval results and system performance.

## Core Methodology
- **Graph-based Retrieval**: Leveraging relationships in a knowledge graph to provide richer context for LLM generation.
- **LightRAG Integration**: Utilizing a lightweight retrieval architecture for reduced latency and high precision.
- **Domain Adaptation**: Fine-tuning via LoRA to improve the model's ability to utilize retrieved technical context.

## Setup
1. Standard environment preparation:
   ```bash
   cd hw2_baseline_code
   pip install -r requirements.txt
   ```
2. Configure `ZHIPUAI_API_KEY` in your environment variables.
3. Execution:
   ```bash
   python run.py
   ```
