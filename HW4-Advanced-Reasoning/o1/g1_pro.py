import time
import re
import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llm.glm.glm_response import glm_response

def analyze_response(messages = None, model = "glm-4-flash", max_tries=10, is_final_answer = False):
    for attempt in range(max_tries):
        try:
            if is_final_answer:
                response = glm_response(messages=messages, model=model)
                # print("response1:", response)
                return response
            else:
                response = glm_response(
                    messages=messages,
                    model=model,
                    response_format={"type": "json_object"}
                )
                # print("response2:", response)
                
                try:
                    parsed_json = json.loads(response)  
                    return parsed_json
                except json.JSONDecodeError:
                    pass  

                json_matches = re.findall(r"```json\n(.*?)\n```", response, re.DOTALL)
                
                if json_matches:
                    parsed_json = json.loads(json_matches[0])
                    return parsed_json

                # raise ValueError("No valid JSON objects found in response")
        except Exception as e:
            if attempt == max_tries - 1:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 5 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 5 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying


def rephrase_question(prompt, model="glm-4-flash"):
    """rStar: A5 rephrase question"""
    messages = [
        {"role": "system", "content": """You are an expert AI assistant skilled at clarifying and refining questions. Your task is to rephrase the given question by stripping away any unnecessary details and keeping only the essential conditions. The goal is to present the question in a more concise, structured, and clear way, while still retaining all key conditions that must be addressed.

        When rephrasing, follow these steps:
        1. Identify the core elements of the question, including any relevant conditions that need to be satisfied.
        2. List these conditions explicitly, using terms like "condition 1," "condition 2," etc., as bullet points.
        3. Eliminate any unnecessary phrasing or irrelevant information that does not directly contribute to the core of the question.

        The rephrased version should focus on the key aspects and conditions while making the problem easier to understand.

        Example of a valid rephrased version:
        Original Question: "How can I calculate the volume of a sphere, given that I know its radius?"
        YOUR RESPONSE SHOULD BE:
        ```
        Rephrased Question:
        - Condition 1: The radius of the sphere is known.
        - Question: What is the formula for calculating the volume of a sphere?
        ```
        """},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now rephrase the question by extracting the essential conditions and presenting them clearly."}
        ]
    
    # rephrase_messages = messages.copy()
    # rephrase_messages.append({"role": "user", "content": "请重新复述以下问题，提炼出关键条件和核心要素：\n" + messages[-1]['content']})
    start_time = time.time()
    
    response = glm_response(
        messages=messages,
        model=model,
        response_format={"type": "text"}
    )
                    
    # rephrased_response = analyze_response(messages, model=model)
    end_time = time.time()
    thinking_time = end_time - start_time
    return response, thinking_time

def split_subquestions(prompt, model="glm-4-flash"):
    """rStar: A3 split subquestions"""
    # 拆分问题并且逐一给出提示，如果问题太过简单就不需要
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that breaks down complex questions into smaller, manageable sub-questions. However, if the question is simple enough to be answered directly, you should repeat the question and provide potential approaches or steps for solving the problem, rather than providing the answer directly.

        Follow these steps:
        1. Assess the complexity of the question. If the question is simple, repeat the question and give potential approaches or steps for solving the problem.
        2. If the question is complex and can be broken down into smaller, more specific aspects, split it into sub-questions.
        3. For each sub-question, provide helpful hints or directions for solving, without giving a direct answer.
        4. After answering the sub-questions, provide a final answer derived from the combined answers. The final answer should only be provided if the sub-questions have all been answered with hints.

        Example of a simple question:
        Original Question: "What is the capital of France?"
        Response: "What is the capital of France? To solve this, think about the most well-known cities in France and consider the one most often associated with politics, culture, and history."

        Example of a complex question:
        Original Question: "How do I calculate the total cost of a trip, including transportation, accommodation, and meals?"
        Rephrased as sub-questions:
        - Sub-question 1: What factors should I consider when calculating the cost of transportation for the trip?
        Hint: Consider the distance, mode of transport, and any other additional costs.
        - Sub-question 2: What should I consider when calculating the cost of accommodation for the trip?
        Hint: Think about the number of nights, the type of accommodation, and potential discounts.
        - Sub-question 3: What factors are involved in calculating the cost of meals for the trip?
        Hint: Take into account the number of meals per day and the average cost of meals in the destination.
        - Final Answer: Only after considering all the sub-questions, combine the factors to get the total trip cost.

        Ensure you evaluate the complexity of the problem and only break it into sub-questions if necessary. If the question is simple, just provide hints and steps to solve it, not the direct answer."""
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now assess the complexity of the question and proceed accordingly—either providing hints for a simple question or splitting it into sub-questions with hints for complex problems."}
    ]
    start_time = time.time()
    # rephrased_response = analyze_response(messages, model=model)
    response = glm_response(
        messages=messages,
        model=model,
        response_format={"type": "text"}
    )
    end_time = time.time()
    thinking_time = end_time - start_time
    return response, thinking_time

def generate_g1_pro_response(prompt, model="glm-4-flash", max_steps=25):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES. 

        Example of a valid JSON response, you [must] follow this format!!, a json format with title, content, next_action:
        ```json
        {
            "title": "Identifying Key Information",
            "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
            "next_action": "continue"
        }```
        """},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    # rephrase the question
    rephrased_response, rephrased_time = rephrase_question(prompt, model=model)
    print(f"Step 0: Rephrased Question\n{rephrased_response} \n")
    steps.append((f"Step 0: Rephrased Question", rephrased_response, rephrased_time))
    # print(type(rephrased_response))
    messages.append({"role": "assistant", "content": rephrased_response})
    
    # split to subquestions
    splited_response, split_time = split_subquestions(prompt + rephrased_response, model=model)
    print(f"Step 0: Splited Question\n{splited_response} \n")
    steps.append((f"Step 0: Splited Question", splited_response, split_time))
    # print(type(splited_response))
    messages.append({"role": "assistant", "content": splited_response})
    
    
    while True:
        start_time = time.time()
        step_data = analyze_response(messages, model=model)
        
        print(f"Step {step_count}: {step_data} \n")
                                    
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        if step_data is not None:
            steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
            messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
            if step_data['next_action'] == 'final_answer': # Maximum of 25 steps to prevent infinite thinking time. Can be adjusted.
                break
        
        if step_count == max_steps:
            break
        step_count += 1


    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."})
    
    start_time = time.time()
    final_data = analyze_response(messages, model=model, max_tries=10, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))

    return steps, total_thinking_time, final_data

if __name__ == '__main__':
    prompt = "how many r in strawberry"
    steps, total_time, final_answer = generate_g1_response(prompt)
    for step in steps:
        print("step:", step)
    print("total_time:", total_time)
    print("final_answer:", final_answer)
    print("\n\n")
