# **Advanced Reasoning in LLM**

## **1. Background**

Advanced reasoning capabilities play a vital role in the application of language models, including multi-step reasoning, logical reasoning, and complex decision-making. OpenAI’s o1 is considered a model with advanced reasoning abilities. Although numerous existing works implement o1-related reasoning algorithms, a comprehensive summary is lacking. This project aims to build a language model system with advanced reasoning capabilities by implementing multiple o1-related algorithms.

## **2. Objective**

Understand and apply advanced reasoning techniques to develop a language model with advanced reasoning capabilities.

1.**Basic Research**: Summarize current methods for enhancing model reasoning capabilities, including but not limited to Test Time Compute, MCTS (Monte Carlo Tree Search), and OpenAI’s O1 model.

2.**Task Definition**: Select appropriate reasoning tasks, starting with basic mathematical reasoning as an example.

3.**Data Preparation**: Prepare relevant datasets, using *math* for training and *math500* for testing.

4.**Model Implementation**:

    •**Model Selection**: Choose and load a suitable language model, such as glm4.

    •**Algorithm Optimization**: Design and optimize algorithms tailored to specific tasks to enhance advanced reasoning capabilities, which may include multi-step reasoning, logical reasoning, and complex decision-making.

5.**Performance Evaluation**:

    •**Reasoning Performance Evaluation**: Evaluate the model’s reasoning performance on the validation set using multiple metrics such as accuracy, reasoning speed, and complexity.

    •**Method Comparison**: Compare the effects of different reasoning methods and analyze results, including comparisons with the base model to demonstrate improvements.

## **3. Implementation Process**

Note: The following implementations are sequential. In practical industrial scenarios, each stage will be optimized for concurrent and pipeline parallel execution to accelerate processing.

1. Basic LLM API implementation: Refer to https://www.bigmodel.cn/dev/api/devguide/sdk-install
2. python==3.10.0
3. pip install -r requirements.txt
4. Obtain your API key from https://bigmodel.cn/usercenter/proj-mgmt/apikeys and place it in llm/glm/api_keys.py (rename the api_keys_temp.py file to api_keys.py). Use the format: glm_key = "***".
5. For basic API calls, refer to llm/glm/glm_response.py. You can replace or supplement other models and implementations (e.g., parallel processing).
6. Dataset Preparation:
7. Download the math dataset: https://people.eecs.berkeley.edu/~hendrycks/MATH.tar, extract it to the data/math directory.
8. Download the math500 dataset: https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/main, place it in the data/math500 directory.
9. Dataset Evaluation Script Preparation:
10. The evaluate.py file in the evaluate folder is used for evaluation and serves as the project’s sole entry point. You can refer to this script to write or call other methods in llm/get_response.py (besides the basic LLM response generation) and compare results. Running python evaluate.py outputs evaluation results and saves the process files.
11. evaluate.py calls two files in the llm folder: get_response.py for solving mathematical problems using the basic model or new methods you implement, and get_critic.py for assessing correctness.
12. Implementation of o1-related algorithms:
13. Create a separate folder for method implementations and provide corresponding interfaces. For example, implement the g1 method (similar to o1) in the o1/g1.py folder.
14. Call these interfaces in llm/get_response.py, assigning a unique identifier for your method to distinguish it from others.

## **4. Basic Results**

First, we tested the performance of the g1 program on a basic problem: “how many r in strawberry,” as shown below:

```
step: ("Step 1: Counting the Occurrences of 'r' in the Word 'Strawberry'", "The word 'strawberry' consists of a total of 9 letters. To find out how many times the letter 'r' appears, we will go through each letter in the word and count the occurrences of 'r'.", 7.770829200744629)

step: ('Step 2: Understanding the Problem', "The problem is asking for the number of times the letter 'r' appears in the word 'strawberry'. This is a straightforward counting problem.", 1.6848220825195312)

step: ("Step 3: Identifying the Occurrences of 'r' in 'Strawberry'", "The letter 'r' is found in the following positions within the word 'strawberry': 2nd, 5th, and 8th. Let's confirm this and count the total occurrences.", 1.7112441062927246)

step: ("Step 4: Counting 'r' in 'Strawberry'", "Now, let's count the occurrences of 'r' in the word 'strawberry'. We find 'r' at positions 2, 3, 5, 7, and 9 in the word.", 2.8369510173797607)

step: ("Step 5: Counting the Occurrences of 'r' in 'Strawberry'", "After identifying the positions of 'r' in 'strawberry', we count that there are 3 occurrences of the letter 'r'.", 2.114763021469116)

step: ('Final Answer', "The final answer is that there are 3 occurrences of the letter 'r' in the word 'strawberry'.", 0.7589890956878662)

total_time: 16.877598524093628

final_answer: The final answer is that there are 3 occurrences of the letter 'r' in the word 'strawberry'.
```

Next, we tested various methods on math500. Here, len indicates how many entries from math500 were used for testing.

| method          | reponse_model | critic_model | len(math500) | math500_accuracy1 | math500_accuracy2 | math500_accuracy3 | acc_average |
| --------------- | ------------- | ------------ | ------------ | ----------------- | ----------------- | ----------------- | ----------- |
| default(0_shot) | glm-4-flash   | glm-4-flash  | 10           | 50%               | 90%               | 90%               | 76.67%      |
| 3_shot          | glm-4-flash   | glm-4-flash  | 10           | 70%               | 70%               | 60%               | 66.67%      |
| 5_shot          | glm-4-flash   | glm-4-flash  | 10           | 70%               | 80%               | 80%               | 76.67%      |
| g1              | glm-4-flash   | glm-4-flash  | 10           | 50%               |                   |                   |             |

## **5. References**

•[Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)

•[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)

•[OpenAI o1 Hub | OpenAI](https://openai.com/o1/)

•[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

•[O1 Replication Journey: A Strategic Progress Report](https://github.com/GAIR-NLP/O1-Journey/blob/main/docs/part2.md)

## **6. Citation**

This code is provided by dujh22 and is under continuous development. Please stay tuned for updates.

```
@misc{dujh22_llmo1_2025,
 author = {Dujh22},
 title = {{LLM-o1: A repository for Large Language Model research and applications}},
 year = {2025},
 howpublished = {\url{https://github.com/dujh22/LLM-o1}},
 note = {GitHub repository}
}
```
