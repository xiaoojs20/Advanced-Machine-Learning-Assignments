# HW4: LLM Advanced Reasoning (o1 Algorithm Implementation)

Project exploring advanced reasoning algorithms such as test-time compute and MCTS for language models.

## Objectives
1. **Algorithm Research**: Summarize and implement techniques to enhance reasoning (e.g., OpenAI o1 logic).
2. **Task**: Focused on mathematical reasoning using the *math* and *math500* datasets.
3. **Implementation**: System based on the GLM-4 model.

## Setup
1. Environment: `python==3.10.0`
2. Dependencies: `pip install -r requirements.txt`
3. Credentials: Set your key in `llm/glm/api_keys.py`.
4. Run Evaluation:
   ```bash
   python evaluate.py
   ```

## Results
Benchmark results on the math500 dataset highlight accuracy improvements using the g1 (o1-like) reasoning strategy compared to standard zero-shot baselines.
