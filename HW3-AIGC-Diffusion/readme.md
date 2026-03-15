# HW3: AIGC - Pokemon Diffusion Fine-tuning

A project focused on artistic style transfer using Stable Diffusion, specifically fine-tuned on a custom Pokemon dataset.

## Project Contents
- **Training Scripts**:
  - `train_text_to_image_lora.py`: Entry point for LoRA-based fine-tuning.
  - `script_better.sh` / `script_baseline.sh`: Optimized shell scripts for multi-node/GPU training.
- **Core Logic**:
  - `diffusers/`: Customized local version of the Hugging Face Diffusers library.
  - `sd_evaluate.py`: Evaluation suite for generated image quality.
  - `mmllm.py`: Multi-modal LLM integration for caption generation.
- **Data**:
  - `pokemon/`: Structured dataset of images and captions.
- **Results**:
  - `fig/`: Visual evidence of style transfer and FID/CLIP score visualizations.

## Implementation Details
- **LoRA Adaptation**: Parameter-efficient fine-tuning concentrating on the UNet cross-attention layers.
- **Multimodal Evaluation**: Using both image-only metrics (FID) and text-image alignment metrics (CLIP Score).
- **RAG Integration**: (Optional) Utilizing `lora_rag_inference.py` for context-aware generation.

## Execution
To start fine-tuning with optimized parameters:
```bash
bash script_better.sh
```
To run visual evaluation:
```bash
python sd_evaluate.py
```