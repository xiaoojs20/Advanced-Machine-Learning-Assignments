# 高级机器学习 (2024-2025) 课程项目集

本仓库收录了高级机器学习课程的全部作业实现与结课大作业。内容涵盖了从大语言模型 (LLM) 的底层加速优化，到复杂的多模态长视频语义理解等多个前沿领域。

## 目录结构

| 模块 | 主题 | 说明与关键技术 |
| --- | --- | --- |
| [**Project**](Project-Video-Understanding/) | **结课大作业** | **长视频场景切割与理解 (Scene-Clipping for Long Video)** |
| [HW1](HW1-LLM-Efficiency/) | LLM 基础 | GLM-4 本地推理与 Flash-Attention 推理加速优化。 |
| [HW2](HW2-RAG-System/) | RAG 与图谱 | 基于 LightRAG 与知识图谱的检索增强生成实现。 |
| [HW3](HW3-AIGC-Diffusion/) | AIGC 扩散模型 | 在宝可梦数据集上通过 LoRA 微调 Stable Diffusion。 |
| [HW4](HW4-Advanced-Reasoning/) | 高级推理 | 语言模型高级推理策略探索 (类 o1 算法实现)。 |
| [Finetuning](HW-Finetuning-Style/) | 参数高效微调 | 基于 LoRA 的情感对齐与特定互联网语体迁移实验。 |

## 技术栈概览
- **深度学习**: Python (PyTorch), Hugging Face Transformers
- **生成式 AI**: Stable Diffusion, Diffusers, LoRA (PEFT)
- **大语言模型**: Llama-2-Chat, GLM-4, Video-LLaMA
- **检索增强**: LightRAG, 向量数据库, Graph-RAG
- **底层优化**: Flash-Attention, 多卡分布式训练脚本

---

## 重点项目：长视频切割与理解
作为本仓库的核心展示项目，该大作业针对长视频内容理解中的时序挑战，提出了场景化分割方案。系统整合了视觉与音频双分支编码器 (ViT-G/14, ImageBind)，并结合 Llama-2 基座模型，实现了对数分钟乃至数十分钟视频的高效语义检索与对答。

**[点击查看详细项目报告](Project-Video-Understanding/README_zh.md)**

---
*Maintainer: xiaojs24*
