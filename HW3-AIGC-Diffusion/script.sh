
# git clone https://github.com/huggingface/diffusers
cd diffusers
# pip install .

cd examples/text_to_image
# pip install -r requirements.txt
# accelerate config

export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli login


export CUDA_VISIBLE_DEVICES=5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="/sddata/finetune/lora/pokemon"


# export DATASET_NAME="lambdalabs/pokemon-blip-captions"
# diffusers/pokemon-gpt4-captions


export HUB_MODEL_ID="pokemon-lora"

## pokemon

export DATASET_NAME="diffusers/pokemon-gpt4-captions"
export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/pokemon_baseline"

accelerate launch  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="no" \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A fierce dragon-type Pokémon breathing fire over a medieval castle." \
  --num_validation_images 4 \
  --seed=1337

## simpsons

# export DATASET_NAME="Norod78/simpsons-blip-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/simpsons_baseline"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="A person eating a donut in the living room, The Simpsons." \
#   --num_validation_images 4 \
#   --seed=1337


## cartoon

export DATASET_NAME="Norod78/cartoon-blip-captions"
export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/cartoon_baseline"

accelerate launch  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="no" \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Two cartoon bears having a picnic under a giant mushroom in a magical forest." \
  --num_validation_images 4 \
  --seed=1337

## rubberduck

# export DATASET_NAME="Norod78/Rubber-Duck-blip-captions"
# export OUTPUT_DIR="/flash2/aml/xiaojs24/AML_3/hw3-base/sddata/finetune/lora/rubberduck_baseline"

# accelerate launch  train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --mixed_precision="no" \
#   --dataset_name=$DATASET_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" \
#   --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --push_to_hub \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --report_to=wandb \
#   --checkpointing_steps=500 \
#   --validation_prompt="A rubber yellow rubber duck floating on a calm lake." \
#   --num_validation_images 4 \
#   --seed=1337

