#!/bin/bash

# =========================================
# Script: train_multiple_tasks.sh
# Description: 通过命令行参数选择并运行不同的数据集任务的 LoRA 微调训练
# Usage: ./train_multiple_tasks.sh [task_name]
# Available Tasks: pokemon, simpsons, cartoon, rubber_duck
# =========================================

# 检查是否提供了任务编号
if [ -z "$1" ]; then
    echo "Usage: $0 [task_number]"
    echo "Available Task Numbers:"
    echo "  1 - Pokémon"
    echo "  2 - Simpsons"
    echo "  3 - Cartoon"
    echo "  4 - Rubber Duck"
    exit 1
fi


TASK=$1

# git clone https://github.com/huggingface/diffusers
cd diffusers
# pip install .

cd examples/text_to_image
# pip install -r requirements.txt
# accelerate config

export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli login



# export DATASET_NAME="lambdalabs/pokemon-blip-captions"
# diffusers/pokemon-gpt4-captions

# 设置通用环境变量
export CUDA_VISIBLE_DEVICES=6
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export LR="1e-05"
export MAX_TRAIN_STEPS=15000
export NUM_TRAIN_EPOCHS=100
export TRAIN_BATCH_SIZE=4
export GRAD_ACC_STEPS=4
export MAX_GRAD_NORM=0.5
export LR_SCHEDULER="cosine"
export LR_WARMUP_STEPS=500
export SNR_GAMMA=5.0
export REPORT_TO="wandb"
export CHECKPOINT_STEPS=500
export NUM_VALIDATION_IMAGES=4
export VALIDATION_EPOCHS=10
export RANK=8
export SEED=1337

# 根据任务编号设置特定的环境变量
case $TASK in
    1)
        # Pokémon 任务
        export CUDA_VISIBLE_DEVICES=6
        export DATASET_NAME="diffusers/pokemon-gpt4-captions"
        export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/pokemon_better"
        export HUB_MODEL_ID="pokemon-lora-better"
        export VALIDATION_PROMPT="A fierce dragon-type Pokémon breathing fire over a medieval castle."
        ;;
    2)
        # Simpsons 任务
        export CUDA_VISIBLE_DEVICES=5
        export DATASET_NAME="Norod78/simpsons-blip-captions"
        export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/simpsons_better"
        export HUB_MODEL_ID="simpsons-lora-better"
        export VALIDATION_PROMPT="A person eating a donut in the living room, The Simpsons."
        ;;
    3)
        # Cartoon 任务
        export CUDA_VISIBLE_DEVICES=4
        export DATASET_NAME="Norod78/cartoon-blip-captions"
        export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/cartoon_better"
        export HUB_MODEL_ID="cartoon-lora-better"
        export VALIDATION_PROMPT="Two cartoon bears having a picnic under a giant mushroom in a magical forest."
        ;;
    4)
        # Rubber Duck 任务
        export CUDA_VISIBLE_DEVICES=3
        export DATASET_NAME="Norod78/Rubber-Duck-blip-captions"
        export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/rubberduck_better"
        export HUB_MODEL_ID="rubber-duck-lora-better"
        export VALIDATION_PROMPT="A bright yellow rubber duck floating on a calm lake."
        ;;
    *)
        echo "Unknown task number: $TASK_NUMBER"
        echo "Available Task Numbers:"
        echo "  1 - Pokémon"
        echo "  2 - Simpsons"
        echo "  3 - Cartoon"
        echo "  4 - Rubber Duck"
        exit 1
        ;;
esac

# 启动训练
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="no" \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=768 --center_crop --random_flip \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --learning_rate=$LR \
  --max_grad_norm=$MAX_GRAD_NORM \
  --lr_scheduler="$LR_SCHEDULER" \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --snr_gamma=$SNR_GAMMA \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=$REPORT_TO \
  --checkpointing_steps=$CHECKPOINT_STEPS \
  --validation_prompt="$VALIDATION_PROMPT" \
  --num_validation_images=$NUM_VALIDATION_IMAGES \
  --validation_epochs=$VALIDATION_EPOCHS \
  --rank=$RANK \
  --seed=$SEED


# ## 1 pokemon

# export DATASET_NAME="diffusers/pokemon-gpt4-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/pokemon_better"
# export HUB_MODEL_ID="pokemon-lora-better"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=768 --center_crop --random_flip \
#   --num_train_epochs=100 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=0.5 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=500 \
#   --snr_gamma=5.0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="A fierce dragon-type Pokémon breathing fire over a medieval castle." \
#   --num_validation_images 4 \
#   --validation_epochs 10\
#   --rank 8 \
#   --seed=1337

# ## 2 simpsons

# export DATASET_NAME="Norod78/simpsons-blip-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/simpsons_better"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=768 --center_crop --random_flip \
#   --num_train_epochs=100 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=0.5 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=500 \
#   --snr_gamma=5.0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="A person eating a donut in the living room, The Simpsons." \
#   --num_validation_images 4 \
#   --validation_epochs 10\
#   --rank 8 \
#   --seed=1337


# ## 3 cartoon

# export DATASET_NAME="Norod78/cartoon-blip-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/cartoon_better"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=768 --center_crop --random_flip \
#   --num_train_epochs=100 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=0.5 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=500 \
#   --snr_gamma=5.0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="Two cartoon bears having a picnic under a giant mushroom in a magical forest." \
#   --num_validation_images 4 \
#   --validation_epochs 10\
#   --rank 8 \
#   --seed=1337

# ## 4 rubberduck

# export DATASET_NAME="Norod78/Rubber-Duck-blip-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/rubberduck_better"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=768 --center_crop --random_flip \
#   --num_train_epochs=100 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-05 \
#   --max_grad_norm=0.5 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=500 \
#   --snr_gamma=5.0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="A rubber yellow rubber duck floating on a calm lake." \
#   --num_validation_images 4 \
#   --validation_epochs 10\
#   --rank 8 \
#   --seed=1337

