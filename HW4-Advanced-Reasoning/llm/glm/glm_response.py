from zhipuai import ZhipuAI

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api_keys import glm_key

client = ZhipuAI(api_key=glm_key)  

def glm_response(prompt = "", messages = None, model = "glm-4-flash", response_format = { "type": "text" }):
    if messages is None:
        try:
            response = client.chat.completions.create(
                model= model,  # glm-4-plus、glm-4-0520、glm-4 、glm-4-air、glm-4-airx、glm-4-long 、 glm-4-flashx 、 glm-4-flash
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4095,
                response_format=response_format
            )
        except Exception as e:
            print(e)
    else:
        try:
            response = client.chat.completions.create(
                model= model,  # glm-4-plus、glm-4-0520、glm-4 、glm-4-air、glm-4-airx、glm-4-long 、 glm-4-flashx 、 glm-4-flash
                messages= messages,
                max_tokens=4095,
            )
        except Exception as e:
            print(e)

    return response.choices[0].message.content

if __name__ == '__main__':
    prompt = "1+1="
    print(glm_response(prompt = prompt))
