# HW4: LLM 高级推理 (类 o1 算法实现)

探索语言模型的高级推理技术，包括测试时计算 (Test-time compute) 与类 o1 算法实现。

## 主要环节
1. **算法研究**: 调研并实现提升模型推理能力的方法（如 MCTS、测试时计算）。
2. **评测任务**: 使用 *math* 与 *math500* 数据集验证数学推理能力。
3. **模型框架**: 以 GLM-4 为基础模型进行算法开发。

## 快速开始
1. 环境: `python==3.10.0`
2. 依赖: `pip install -r requirements.txt`
3. 配置: 在 `llm/glm/api_keys.py` 中设置 API Key。
4. 评测:
   ```bash
   python evaluate.py
   ```

## 实验结果
在 math500 数据集上的测试表明，通过 g1 推理策略可以有效提升模型的逻辑推导准确率。
