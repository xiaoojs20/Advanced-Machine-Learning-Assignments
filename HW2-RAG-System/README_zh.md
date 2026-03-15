# HW2: 检索增强生成 (RAG)

基于 LightRAG 与知识图谱技术的高级检索增强生成流水线实现。

## 项目结构
- **hw2_baseline_code/**: RAG 系统的主体实现，集成了 LightRAG 框架。
- **database/**: 本地文档存储及已生成的向量/图索引数据。
- **src/**: 包含向量嵌入生成与查询处理的辅助脚本。
- **lora/**: 用于使模型更好适应检索任务的适配器权重。
- **fig/**: 检索结果与系统性能的可视化分析。

## 核心技术
- **图检索 (Graph-based Retrieval)**: 利用知识图谱中的实体关系，为 LLM 生成提供更丰富的上下文。
- **LightRAG 集成**: 采用轻量化检索架构，平衡响应延迟与检索精度。
- **领域适配**: 通过 LoRA 微调提升模型对专业领域检索内容的理解与利用率。

## 运行指导
1. 环境准备：
   ```bash
   cd hw2_baseline_code
   pip install -r requirements.txt
   ```
2. 在环境变量中设置 `ZHIPUAI_API_KEY`。
3. 运行：
   ```bash
   python run.py
   ```
