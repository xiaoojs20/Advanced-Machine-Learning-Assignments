import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
from llm.get_response import get_answer
from llm.get_critic import get_critic
from tqdm import tqdm
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

save_path = "/flash2/aml/xiaojs24/AML_4/evaluate/results"
test_dataset = "math"


def get_last_sentence(s):
    # use \n to split string
    paragraphs = s.split('\n')
    last_paragraph = None
    
    # from back to front, find the first non-empty paragraph
    for paragraph in reversed(paragraphs):
        if paragraph.strip():
            last_paragraph = paragraph
            break
    
    if not last_paragraph: # if the last paragraph is empty, return an empty string
        return ""
    
    # if the last paragraph is not empty, use \n to split it
    sentences = re.split(r'(?<=[。！？.?])', last_paragraph)
    last_sentence = None
    
    # from back to front, find the first non-empty sentence
    for sentence in reversed(sentences):
        if sentence.strip():
            last_sentence = sentence
            break
    
    return last_sentence.strip() if last_sentence else last_paragraph.strip()

def evaluate_one(method, data, response_model, critic_model):
    prediction = ""
    correct = False

    prediction = get_answer(data["problem"], data["subject"], data["level"], method, response_model)
    # get the last sentence of the prediction as the answer
    prediction_answer = get_last_sentence(prediction)

    correct = get_critic(
        question = data["problem"],
        model = critic_model,
        solution = data["solution"], 
        answer = data["answer"], 
        response = prediction, 
        response_answer = prediction_answer
    )
    
    return prediction, prediction_answer, correct

from tqdm import tqdm
import json

def evaluate(label_sentence, method, data_path="../data/math500",response_model="glm-4-flash", critic_model="glm-4-flash"):
    all_count = 0
    correct_count = 0
    with open(os.path.join(os.path.join(save_path, test_dataset), f"evaluate_result_{label_sentence}.jsonl"), "w") as f_out:
        # with open("../data/math500/test.jsonl", "r") as f:
        with open(os.path.join(data_path, "test.jsonl"), "r") as f:
            lines = f.readlines()
            for line in tqdm(
                lines,
                desc=f"Evaluating {label_sentence}",
                unit="line",
                colour="blue",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}"
            ):
                data = json.loads(line)

                temp_prediction, temp_prediction_answer, temp_correct = evaluate_one(method, data, response_model, critic_model)
                all_count += 1
                if temp_correct:
                    correct_count += 1

                temp = {
                    "ID": data["unique_id"],
                    "Correct": temp_correct,
                    "Predicted Answer": temp_prediction_answer,
                    "Predicted": temp_prediction,
                }
                f_out.write(json.dumps(temp) + "\n")
    
    # 打印正确率
    print(f"Correct rate: {correct_count / all_count:.2%}")
    return correct_count / all_count

if __name__ == "__main__":
    method_list = ["default", "3shot", "5shot", "g1", "g1_pro"]
    label_list = ["default",  "3shot", "5shot", "g1", "g1_pro"]
    
    label_sentence = "g1"
    method = "g1" # defalut, 3shot, 5shot, g1
    response_model = "glm-4-flash"
    critic_model = "glm-4-flash"
    
    evaluate_try_times = 3

    # for method in method_list:
    for method in ["g1_pro"]:
        auc_list = []        
        print(f"Method: {method}")
        for try_time in range(evaluate_try_times):
            label_sentence = method + f"_{try_time}"
            auc = evaluate(label_sentence, method, data_path="../data/hard/"+test_dataset)
            # "../data/math500"
            # "../data/hard/physics"
            # "../data/hard/math"
            auc_list.append(auc)
        print(f"Average AUC: {sum(auc_list) / len(auc_list):.2%}")        
        
        # save auc_list to a file
        save_folder = os.path.join(save_path, test_dataset)
        # with open(f"evaluate_auc.json", "a") as f: 
        with open(os.path.join(save_folder, f"evaluate_auc.jsonl"), "a") as f:
            temp = {
                "Method": method,
                "AUC": auc_list,
                "Average AUC": sum(auc_list) / len(auc_list),
            }
            f.write(json.dumps(temp) + "\n")

            