import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
from video_llama.datasets.datasets.video_instruct_dataset import preprocess_multimodal, preprocess_for_llama_v2, convert_source_vicuna_format, preprocess
import pandas as pd
import copy
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.processors import transforms_video,AlproVideoTrainProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from typing import Dict, Optional, Sequence, List
import json

caption_prompt = [
    "Describe the given video in detail.",
    "Elaborate on the video's content.",
    "Provide a detailed explanation of the video.",
    "Explain the video thoroughly.",
    "Give an in-depth description of the video."
]

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
class ShareGPT4VideoDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root, num_video_query_token=32,tokenizer_name = '/flash2/aml/public/models/vicuna-7b-delta-v0',data_type = 'video', model_type='vicuna'):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        with open(ann_root, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        self.annotation = data

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 32
        self.frm_sampling_strategy = 'headtail'

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type

    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root,  sample['video_path'])
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample_dict = self.annotation[index]
                video_id = sample_dict['video_id']

                if 'captions' in sample_dict.keys():
                    text = ""
                    for cap in sample_dict['captions']:
                        text += cap['content']
                else:
                    raise NotImplementedError("Un-supported text annotation format.")

                # fetch video
                video_path = self._get_video_path(sample_dict) 
                conversation_list = self.text_preprocess(text)
                # video = self.vis_processor(video_path)
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling ="uniform", return_msg = True
                )
                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                    # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=self.num_video_query_token,msg = msg)
                new_sources = convert_source_vicuna_format(sources)
                
                if self.model_type =='vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                else:
                    print('not support')
                    raise('not support')
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)
    
    def text_preprocess(self, text) -> List[Dict[str, str]]:
        caption = text

        conversations = [
            {
                'q': random.choice(caption_prompt),
                'a': caption
            }
        ]

        return conversations
    
    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch
    
    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result
        
