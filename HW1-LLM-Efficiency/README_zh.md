# HW1: LLM 基础与效率优化

大语言模型 (LLM) 核心组件的实现以及基于 Flash-Attention 的推理性能优化。

## 项目结构
- **src/**: 核心代码。
  - `run_glm4.py`: 基于智谱 AI SDK 的 GLM-4 模型本地推理入口。
- **flash_attn/**: Flash-Attention 机制的集成实现，旨在加速计算并降低显存占用。
- **server_code/**: 用于服务器端部署及远程调用的脚本。
- **submit/**: 包含作业报告及验证结果。

## 关键特性
- **本地化部署**: 掌握 9B 级别模型在本地或专用服务器环境下的部署流程。
- **内存优化**: 通过 Flash-Attention 优化自注意力层，显著提升长文本处理能力。
- **配置管理**: 通过 `generation_config.json` 灵活调整采样策略与生成参数。

## 运行说明
1. 确保环境已正确安装 `flash-attention` 及其依赖项。
2. 在 `src/run_glm4.py` 中配置相应的 API Key。
3. 启动推理：
   ```bash
   python src/run_glm4.py
   ```
