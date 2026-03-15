# Finetuning Homework: LLM Style Transfer & Alignment

A study on Parameter-Efficient Fine-Tuning (PEFT) using LoRA to adapt LLMs for specific emotional tones and internet-based linguistic styles.

## Project Structure
- **src/**: Data processing scripts and the primary adaptation notebook `data_adapter.ipynb`.
- **data/**: Curated datasets for style transfer training.
- **微调结果/**: Collection of comparative results demonstrating model behavior before and after fine-tuning.

## Experimental Objectives
- **Stylistic Adaptation**: Training the model to adopt internet-specific slangs and cultural tones (e.g., "B-station style").
- **Persona Alignment**: Aligning output with specific personality traits and user emotional states.

## Methodology
- **LoRA (Low-Rank Adaptation)**: Efficiently updating a small fraction of model weights while keeping the base LLM frozen.
- **PEFT Integration**: Leveraging the Hugging Face PEFT library for robust and reproducible adapter management.
