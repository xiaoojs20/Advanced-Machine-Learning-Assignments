# HW1: LLM Fundamentals & Efficiency Optimization

Core implementations of Large Language Model (LLM) components and performance optimization using Flash-Attention.

## Project Structure
- **src/**: Core logic scripts.
  - `run_glm4.py`: Execution entry point for GLM-4 inference using the Zhipu AI SDK.
- **flash_attn/**: Implementation of the Flash-Attention mechanism for accelerated computation and reduced memory footprint.
- **server_code/**: Scripts for server-side deployment and remote inference.
- **submit/**: Contains formal reports and validation results.

## Key Features
- **Local Deployment**: Practical setup for running 9B+ parameter models on local/server environments.
- **Memory Efficiency**: Integration of Flash-Attention to optimize the self-attention layer.
- **Configuration Management**: Parameter tuning via `generation_config.json` for customized text generation behavior.

## Setup & Execution
1. Ensure the `flash-attention` library and dependencies are correctly installed.
2. Configure API credentials in `src/run_glm4.py`.
3. Start inference:
   ```bash
   python src/run_glm4.py
   ```
