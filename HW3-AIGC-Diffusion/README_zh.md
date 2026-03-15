# HW3: AIGC - 宝可梦扩散模型微调

本项目专注于使用 Stable Diffusion 进行艺术风格迁移，特别是在自定义宝可梦数据集上的微调实现。

## 项目内容
- **训练脚本**:
  - `train_text_to_image_lora.py`: 基于 LoRA 的微调程序。
  - `script_better.sh` / `script_baseline.sh`: 用于多 GPU 训练的优化脚本。
- **核心逻辑**:
  - `diffusers/`: 定制化的本地 `diffusers` 库实现。
  - `sd_evaluate.py`: 生成图像质量的评估套件。
  - `mmllm.py`: 用于生成文本描述的多模态模型集成。
- **数据**:
  - `pokemon/`: 包含图像与对应标注的结构化数据集。
- **结果记录**:
  - `fig/`: 风格迁移效果及 FID/CLIP 指标的可视化展示。

## 实现细节
- **LoRA 适配**: 对预训练 Stable Diffusion 模型的 UNet 交叉注意力层进行参数高效微调。
- **多维度评估**: 结合图像质量指标 (FID) 与图文对齐指标 (CLIP Score)。
- **(可选) RAG 集成**: 通过 `lora_rag_inference.py` 实现基于检索增强的内容生成。

## 运行指导
使用优化参数启动微调：
```bash
bash script_better.sh
```
执行可视化评估：
```bash
python sd_evaluate.py
```
