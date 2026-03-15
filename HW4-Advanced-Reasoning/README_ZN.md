# **LLM的高级推理**

## 1. 背景

高级推理能力在语言模型应用中占有非常重要的地位，包括应用于多步推理、逻辑推理和复杂决策等方面。OpenAI的o1被认为是一个具备高级推理能力的模型。尽管有许多现成的工作对o1相关推理算法进行了实现，但缺乏较为全面的总结。本项目通过实现多个o1相关的算法，以构造一个具有高级推理能力的语言模型系统。

## 2. **目标**

理解和应用高级推理技术，实现一个具有高级推理能力的语言模型。

1. **基本调研** ：汇总目前用于提高模型推理能力的各种方法，包括但不限于Test Time Compute（测试时计算）、MCTS（蒙特卡洛树搜索）、OpenAI's O1模型等。
2. **任务定义** ：选择合适的推理任务。这里首先以数学的基础推理为例。
3. **数据准备** ：准备相关的数据集，我们采用math用来训练，math500进行测试。
4. 模型实现 ：
   - **模型选择** ：选择并加载一个合适的语言模型，如glm4。
   - **算法优化** ：根据具体任务的需求设计和优化算法，以提升模型的高级推理能力。这可能包括多步推理、逻辑推理和复杂决策等。
5. 性能评估 ：
   - **推理性能评估** ：在验证集上评估模型的推理性能。应使用多种评估指标，如准确性、推理速度和复杂度。
   - **方法比较** ：比较不同推理方法的效果，分析结果。包括与基础版本的模型进行对比，以显示优化的效果。

## 3. **具体实现过程**

注意，以下实现均为串行实现，在实际工业场景中，会对各阶段进行并发和流水线并行优化加速。

1. LLM基本调用实现，参照https://www.bigmodel.cn/dev/api/devguide/sdk-install完成
   1. python==3.10.0
   2. pip install -r requirements.txt
   3. 你需要在https://bigmodel.cn/usercenter/proj-mgmt/apikeys找到你的key，并按照如下格式放到llm/glm/api_keys.py(你需要将api_keys_temp.py文件重命名为api_keys.pyllm/glm/api_keys.py)中：glm_key = "***"
   4. 基本api调用实现详见 llm/glm/glm_response.py, 你可以更换或者补充其他模型以及实现方式（并行并发等）
2. 数据集准备
   1. math数据集下载：https://people.eecs.berkeley.edu/~hendrycks/MATH.tar,下载后解压缩到data\math路径下
   2. math500数据集下载：https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/main，下载后放到data\math500路径下
3. 数据集评估脚本准备：
   1. 你可以直接在evaluate文件夹下找到evaluate.py文件，该文件用于进行评估，也是整个项目的唯一入口，你可参照此脚本中的调用逻辑在该llm\get_response.py下写或者调用其他方法（除去LLM基本调用生成回复的方法外任何类似o1的方法），并将两者的结果进行对比（python evaluate.py结束后会输出对应的评估结果，并保存过程文件）
   2. evaluate.py文件会调用llm文件夹中的两个文件，get_response.py文件用于调用llm中基础模型的response函数进行数学问题的解答（或者调用你写的新方法进行解答），get_critic.py文件用于调用llm中不同的基础模型的、对应的response函数进行数学问题解答正确性的评估
4. o1相关算法的实现：
   1. 你可以单独建立文件夹进行具体方法的实现，并提供对应的接口，比如这里实现了类似o1的方式g1在o1\g1.py文件夹下，具体效果如下
   2. 该接口在llm/get_response.py中被调用，你应该给你的方法method命名一个对应的标识，以和不同的方法进行区分

## 4. 基本结果

首先，我们测试了g1程序在基本问题“how many r in strawberry"上的表现，如下：

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

然后我们测试了math500上各种方法的表现，这里len表示选择math500的前多少条数据进行测试

| method          | reponse_model | critic_model | len(math500) | math500_accuracy1 | math500_accuracy2 | math500_accuracy3 | acc_average |
| --------------- | ------------- | ------------ | ------------ | ----------------- | ----------------- | ----------------- | ----------- |
| default(0_shot) | glm-4-flash   | glm-4-flash  | 10           | 50%               | 90%               | 90%               | 76.67%      |
| 3_shot          | glm-4-flash   | glm-4-flash  | 10           | 70%               | 70%               | 60%               | 66.67%      |
| 5_shot          | glm-4-flash   | glm-4-flash  | 10           | 70%               | 80%               | 80%               | 76.67%      |
| g1              | glm-4-flash   | glm-4-flash  | 10           | 50%               |                   |                   |             |

## **5. 参考资料**

- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [OpenAI o1 Hub | OpenAI](https://openai.com/o1/)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [O1 Replication Journey: A Strategic Progress Report](https://github.com/GAIR-NLP/O1-Journey/blob/main/docs/part2.md)

## 6. 引用

本代码由dujh22提供，并且处于动态更新中，请持续关注包更新。

```
@misc{dujh22_llmo1_2025,
  author = {Dujh22},
  title = {{LLM-o1: A repository for Large Language Model research and applications}},
  year = {2025},
  howpublished = {\url{https://github.com/dujh22/LLM-o1}},
  note = {GitHub repository}
}
```
