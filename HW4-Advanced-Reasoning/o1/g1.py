import time
import re
import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llm.glm.glm_response import glm_response

def analyze_response(messages = None, model = "glm-4-flash", is_final_answer = False):
    for attempt in range(5):
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
            if attempt == 4:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 5 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 5 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_g1_response(prompt, model="glm-4-flash"):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
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
        
        if step_count == 25:
            break
        step_count += 1


    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. Do not use JSON formatting. Only provide the text response without any titles or preambles. Retain any formatting as instructed by the original prompt, such as exact formatting for free response or multiple choice."})
    
    start_time = time.time()
    final_data = analyze_response(messages, model=model, is_final_answer=True)
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
