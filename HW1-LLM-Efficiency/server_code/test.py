import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math
from torch import nn
import torch.nn.functional as F
import subprocess
import time
import numpy as np
import pandas as pd
import random

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import GenerationConfig

print(torch.__version__)
print(torch.version.cuda)  # 检查 PyTorch 使用的 CUDA 版本
print(torch._C._GLIBCXX_USE_CXX11_ABI)


from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def warmup_and_benchmark(
    model,
    tokenizer,
    max_seq_len,
    num_batches,
    max_new_tokens,
):
    inputs = tokenizer("Hi" * max_seq_len, return_tensors="pt").to("cuda")

    # warmup
    # _ = model.generate(
    #     **inputs,
    #     max_new_tokens=20,
    #     pad_token_id=tokenizer.eos_token_id,
    #     use_cache=False,
    # )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_max_memory_allocated()
    memory_before = torch.cuda.memory_allocated() / 1e6  # Convert to MB
    
    with torch.no_grad():
        start_event.record()
        torch.cuda.empty_cache()
        for _ in range(num_batches):
            _ = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        end_event.record()
        torch.cuda.synchronize()

    memory_after = torch.cuda.memory_allocated()
    memory_after_max = torch.cuda.max_memory_allocated()/ 1e6  # Convert to MB
    memory_usage = (memory_after_max - memory_before) 
    forward_timing = (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches
    

    return forward_timing, memory_usage, memory_after_max



    
def run_glm4(tokenizer, query, model, generation_config, test_config, device):
    model = model.to(device)
    model.cuda()  # Move model to GPU
    # print(model.state_dict().keys())

    # query -> input
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )
    # -> parallel
    # inputs = inputs.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    num_tokens = inputs['input_ids'].shape[1]
    
    # run model
    # print(model)
    
    # scale_list = [1, 10, 100, 500, 1000]
    if test_config.test_times > 1:
        # pass
        time_used = 0
        memory_used = 0
        num_tokens_list = 0

        for _ in range(test_config.test_times):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(device=device) # 确保cuda操作完成
            start_time = time.time()
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
                with torch.no_grad():
                    # parallel: model.generate -> model.module.generate
                    outputs = model.generate(**inputs, generation_config=generation_config)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    # print(outputs[0])
                    # print("Outputs: ", end="")
                    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            torch.cuda.synchronize(device=device) # 再次确保cuda操作完成
            end_time = time.time()
            # scale tokens 下，单次推理耗时
            num_tokens_list = num_tokens
            time_used += (end_time - start_time) / test_config.test_times # sec
            memory_used += torch.cuda.max_memory_allocated() / (1024 ** 2) / test_config.test_times  # 转为 MB
            
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(device=device)
        start_time = time.time()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            with torch.no_grad():
                # parallel: model.generate -> model.module.generate
                outputs = model.generate(**inputs, generation_config=generation_config)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                # print(outputs[0])
                # print("Outputs: ", end="")
                # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        torch.cuda.synchronize(device=device)
        end_time = time.time()
        num_tokens_list = num_tokens
        time_used = end_time - start_time # sec
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转为 MB
        print("Outputs: ", end="")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
    return num_tokens_list, time_used, memory_used


if __name__ == "__main__":
    torch.cuda.set_device(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cuda device count: {torch.cuda.device_count()}, device: {device}")
    random.seed(42)
    
    num_batches = 1
    max_seq_len = 512
    max_batch_size = 512
    max_new_tokens = 512

    MODEL_PATH = "THUDM/glm-4-9b-chat"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initial input text
    input_text = "Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest."
    input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
    
    
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model_fa_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs).eval().cuda().to(device)
    model_fa = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_fa_kwargs).eval().cuda().to(device)
    # "flash_attention_2"
    # model.cuda()  # Move model to GPU
    # model.to(device)
    
    
    native_total_time_dict = {}
    fa2_total_time_dict = {}
    forward_speedups = {}
    native_total_memory_dict = {}
    fa2_total_memory_dict = {}
    native_total_memory_max_dict = {}
    fa2_total_memory_max_dict = {}
    memory_speedups = {}
    memory_max_speedups = {}
    for max_seq_len in [256, 1024, 2048, 4096]:
        print(f"Running for sequence length {max_seq_len}")
        native_timing, native_momory, native_memory_max = warmup_and_benchmark(
            model,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        native_total_time_dict[f"{max_seq_len}"] = native_timing
        native_total_memory_dict[f"{max_seq_len}"] = native_momory
        native_total_memory_max_dict[f"{max_seq_len}"] = native_memory_max
        

        fa2_timing, fa2_memory, fa2_memory_max = warmup_and_benchmark(
            model_fa,
            tokenizer,
            max_seq_len,
            num_batches,
            max_new_tokens,
        )
        fa2_total_time_dict[f"{max_seq_len}"] = fa2_timing
        fa2_total_memory_dict[f"{max_seq_len}"] = fa2_memory
        fa2_total_memory_max_dict[f"{max_seq_len}"] = fa2_memory_max

        forward_speedups[f"{max_seq_len}"] = native_timing / fa2_timing
        memory_speedups[f"{max_seq_len}"] = native_momory / fa2_memory
        memory_max_speedups[f"{max_seq_len}"] = native_memory_max / fa2_memory_max

        print(f"Native total time: {native_timing:.2f}s")
        print(f"FA2 total time: {fa2_timing:.2f}s")
        print(f"Speedup: {forward_speedups[f'{max_seq_len}']:.2f}x")
        print(f"Native total memory: {native_momory:.2f}MB")
        print(f"FA2 total memory: {fa2_memory:.2f}MB")
        print(f"Memory speedup: {memory_speedups[f'{max_seq_len}']:.2f}x")
        print(f"Native total memory max: {native_memory_max:.2f}MB")
        print(f"FA2 total memory max: {fa2_memory_max:.2f}MB")
        print(f"Memory max speedup: {memory_max_speedups[f'{max_seq_len}']:.2f}x")
    
    # plt.figure(figsize=(10, 6)) 
    # sns.set(style="darkgrid")
    # # plot both lines
    # sns.lineplot(data=native_total_time_dict, color="blue", label=f"eager")
    # sns.lineplot(data=fa2_total_time_dict, color="orange", label=f"flash_attention_2")
    # plt.ylabel("Average inference time (s)")
    # plt.xlabel("Seq Length")
    # plt.title(
    #     "Comparing average inference time between standard vs Flash Attention-2",
    #     fontsize=8,
    # )
    # plt.legend()
    # # save plot
    # plt.savefig("./timing_plot.jpg", dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(10, 6)) 
    # sns.set(style="darkgrid")
    # sns.lineplot(data=native_total_memory_dict, color="blue", label=f"eager")
    # sns.lineplot(data=fa2_total_memory_dict, color="orange", label=f"flash_attention_2")
    # plt.ylabel("Average memory usage (MB)")
    # plt.xlabel("Seq Length")
    # plt.title(
    #     "Comparing average memory usage between standard vs Flash Attention-2",
    #     fontsize=8,
    # )
    # plt.legend()
    # # save plot
    # plt.savefig("./memory_plot.jpg", dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(10, 6))
    # sns.set(style="darkgrid")
    # sns.lineplot(data=native_total_memory_max_dict, color="blue", label=f"eager")
    # sns.lineplot(data=fa2_total_memory_max_dict, color="orange", label=f"flash_attention_2")
    # plt.ylabel("Max memory usage (MB)")
    # plt.xlabel("Seq Length")
    # plt.title(
    #     "Comparing max memory usage between standard vs Flash Attention-2",
    #     fontsize=8,
    # )
    # plt.legend()
    # # save plot
    # plt.savefig("./memory_max_plot.jpg", dpi=300)
    # plt.close()
    
    
    
    # generation_config = GenerationConfig(
    #     eos_token_id=[151329,151336,151338],
    #     pad_token_id= 151329,
    #     do_sample= True,
    #     # do_sample= False, # for test flashattn 禁用随机性，确保生成结果一致
    #     temperature= 0.8,
    #     max_length= 100000,
    #     min_new_tokens= 2000,
    #     max_new_tokens= 2000+10,
    #     top_p= 0.8,
    #     top_k= 1,
    #     # top_k=None,            # 不限制候选集大小
    #     # top_p=1.0,             # 使用完整的分布生成
    #     transformers_version= "4.44.0")
    
    # class TestConfig:
    #     def __init__(self):
    #         self.test_times = 10
    #         # self.input_scale_list = [1, 10, 100, 500, 1000]
    #         self.input_scale_list = [10]
    #         self.output_length_list = [10, 200] #  1000] # 2000, 5000, 10000]
    #         # self.output_length_list = [2000, 5000]

    # test_config = TestConfig()
    
    # num_tokens_list = np.zeros(len(test_config.output_length_list))
    # avg_time_used = np.zeros(len(test_config.output_length_list))
    # avg_memory_used = np.zeros(len(test_config.output_length_list))
    
    # # attn_type, num_tokens, time_used, memory_used = run_glm4(tokenizer, base_query, model, generation_config, test_config, device)
    # base_query = "Who are you?" * 10
        
    # for i, output_length in enumerate(test_config.output_length_list):
    #     generation_config.min_new_tokens = output_length
    #     generation_config.max_new_tokens = output_length + 5
    #     print(f"Test output_length: {output_length}")
    #     num_tokens, time_used, memory_used = run_glm4(tokenizer, base_query, model, generation_config, test_config, device)
        
    #     num_tokens_list[i] = num_tokens
    #     avg_time_used[i] = time_used
    #     avg_memory_used[i] = memory_used
    
    # print(f"num_tokens_list: {num_tokens_list}")
    # print(f"avg_time_used: {avg_time_used}")
    # print(f"avg_memory_used: {avg_memory_used}")
        
    