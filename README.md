# Advanced Machine Learning (2024-2025)

This repository contains a collection of assignments and a final project from the Advanced Machine Learning course. The work covers a broad spectrum of modern AI, ranging from the fundamental mechanics of Large Language Models to complex multi-modal video understanding.

## Repository Structure

| Module | Topic | Description & Key Tech |
| --- | --- | --- |
| [**Project**](Project-Video-Understanding/) | **Final Project** | **Scene-Clipping Long Video For Better Understanding.** (Video-LLaMA) |
| [HW1](HW1-LLM-Efficiency/) | LLM Core | Efficient inference and Flash-Attention implementation on GLM-4. |
| [HW2](HW2-RAG-System/) | RAG & Graphs | Retrieval-Augmented Generation using LightRAG and Knowledge Graphs. |
| [HW3](HW3-AIGC-Diffusion/) | AIGC | Fine-tuning Stable Diffusion on a Pokemon dataset with LoRA. |
| [HW4](HW4-Advanced-Reasoning/) | LLM Reasoning | Exploration of advanced reasoning strategies (类 o1 algorithms). |
| [Finetuning](HW-Finetuning-Style/) | PEFT | Tone and style alignment experiments via LoRA adapters. |

## Technical Stack
- **Languages**: Python (PyTorch), SQL, Shell
- **Generative AI**: Stable Diffusion, Diffusers, LoRA (PEFT)
- **Large Language Models**: Llama-2-Chat, GLM-4, Video-LLaMA
- **Retrieval**: LightRAG, Vector DBs, Graph-based RAG
- **Hardware Optimization**: Flash-Attention, Multi-GPU training scripts

---

## Featured Project: Scene-Clipping for Long Videos
The showcase of this repository is the Final Project, where we addressed the challenge of processing long-form videos by decomposing them into semantic scenes. By integrating visual and audio encoders (ViT-G/14, ImageBind) with a Llama-2 backbone, the system can answer complex queries about specific events within lengthy videos.

**[View Detailed Project Report](Project-Video-Understanding/README.md)**

---
*Created and maintained by xiaojs24*
