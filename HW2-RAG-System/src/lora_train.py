import torch
import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, EarlyStoppingCallback
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# load data
df = pd.read_json(config['data']['QA'])
df_train, df_valid = train_test_split(df, test_size=0.15, random_state=config['seed'])
ds_train = Dataset.from_pandas(df_train)
ds_valid = Dataset.from_pandas(df_valid)

print(ds_train.shape)
print(ds_train[:3])


def process_func(example, tokenizer, max_length=512):
    """
    QA 数据集预处理

    Args:
        example (dict): 包含 question 和 answer 的字典
        tokenizer (Tokenizer)
        max_length (int):
    Returns:
        dict: 包含 input_ids, attention_mask, 和 labels 的字典。
    """
    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    encoded_instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n你是一个机器学习、深度学习的专家，你会接收到一个相关主题的问题，请输出该问题的正确答案<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        add_special_tokens=False
        )  
    # add_special_tokens 不在开头加 special_tokens
    encoded_response = tokenizer(
        f"{example['answer']}<|eot_id|>", 
        add_special_tokens=False
        )
    
    input_ids = encoded_instruction["input_ids"] + encoded_response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = encoded_instruction["attention_mask"] + encoded_response["attention_mask"] + [1]  
    # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(encoded_instruction["input_ids"]) + encoded_response["input_ids"] + [tokenizer.pad_token_id]  
    # print(len(input_ids), len(attention_mask), len(labels)) ~200
    if len(input_ids) > max_length:  # 做一个截断
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def main():
    if config['lora']['is_wandb']:
            wandb.init(
                project="LoRA-GLM4-9B",    # 您的 wandb 项目名
                name="lora-finetune-run",  # 此次运行名称，可自行定义
            )
    # tokenize / encode the dataset
    tokenizer = AutoTokenizer.from_pretrained(
                    config['model']['glm4-9b'], 
                    use_fast=False, 
                    trust_remote_code=True
                    )
    tokenizer.pad_token = tokenizer.eos_token
    def map_fn(examples):
        return process_func(examples, tokenizer, max_length=512)
    tokenized_id_train = ds_train.map(map_fn, remove_columns=ds_train.column_names)
    tokenized_id_valid = ds_valid.map(map_fn, remove_columns=ds_valid.column_names)


    # load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['glm4-9b'], 
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        )
    model = model.to(device)
    model.enable_input_require_grads()
    print(model)

    ## LoRA config
    loraconfig = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=config['lora']['target_modules'],
        r=config['lora']['r'],                          # Lora 秩 8
        lora_alpha=config['lora']['lora_alpha'],        # Lora alaph 32
        lora_dropout=config['lora']['lora_dropout'],    # Dropout 0.01
        inference_mode=False                            # 训练模式
    )

    model = get_peft_model(model, loraconfig)
    model.print_trainable_parameters()

    ## lora train
    args = TrainingArguments(
        output_dir=config['lora']['path'],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=config['lora']['num_train_epochs'],
        weight_decay=0.01,
        learning_rate=1e-4,
        save_on_each_node=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=100,
        run_name="GLM4-9B-lora-wandb", # 也可指定 run_name
        report_to=["wandb"],           # 让HF Trainer把日志上报到 wandb
        load_best_model_at_end=True,
        gradient_checkpointing=True
        # metric_for_best_model="eval_auc",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id_train,
        eval_dataset=tokenized_id_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 启用早停
    )

    trainer.train()
    trainer.save_model(config['lora']['path'])
    print(f"LoRA model saved to: {config['lora']['path']}")
    
    if config['lora']['is_wandb']:
        wandb.finish()

if __name__ == '__main__':
    main()
