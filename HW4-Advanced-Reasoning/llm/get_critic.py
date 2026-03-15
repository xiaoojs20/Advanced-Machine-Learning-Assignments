from get_response import get_response
import re

PROMPT_TEMPLATE = '''

[Instruction]\n请你扮演一名评审，对人工智能助手对数学问题生成的回答进行评价。你的评价需要优先考虑最终答案的正确性，其次关注解题步骤每一步的正确性。最后，回答正确的评价为1，回答错误的评价为0。
我会提供一段参考答案和人工智能助手的答案，请你尽可能客观地评价。请你输出 1) 首先提供一段简短的解释，用来评价人工智能助手回答的质量，如有最终答案错误或者步骤错误，请指出并简单解释；2) 然后给出评估结果，必须严格按照以下格式进行评价：\"[[rating]]\"，例如：\"评分:[[1]]\".

[Question]
{problem}

[The Start of Reference Answer]
{reference_answer}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{assistant_ansiwer}
[The End of Assistant's Answer]

'''

def get_critic(question, solution, answer, response, response_answer, model = "glm-4-flash"):
    query1 = PROMPT_TEMPLATE.format(
        problem=question,
        reference_answer=solution,
        assistant_ansiwer=response
    )
    critic1 = get_response(query1, model)
    rating1 = re.findall(r"\[\[(\d+)\]\]", critic1)
    query2 = PROMPT_TEMPLATE.format(
        problem=question,
        reference_answer=solution,
        assistant_ansiwer=response_answer
    )
    critic2 = get_response(query2, model)
    rating2 = re.findall(r"\[\[(\d+)\]\]", critic2)
    query3 = PROMPT_TEMPLATE.format(
        problem=question,
        reference_answer=answer,
        assistant_ansiwer=response
    )
    critic3 = get_response(query3, model)
    rating3 = re.findall(r"\[\[(\d+)\]\]", critic3)
    query4 = PROMPT_TEMPLATE.format(
        problem=question,
        reference_answer=answer,
        assistant_ansiwer=response_answer
    )
    critic4 = get_response(query4, model)
    rating4 = re.findall(r"\[\[(\d+)\]\]", critic4)

    # print(f"rating1: {rating1}")
    # print(f"rating2: {rating2}")
    # print(f"rating3: {rating3}")
    # print(f"rating4: {rating4}")
    # print(f"critic1: {critic1}")
    # print(f"critic2: {critic2}")
    # print(f"critic3: {critic3}")
    # print(f"critic4: {critic4}")
    # print(f"query1: {query1}")
    # print(f"query2: {query2}")
    # print(f"query3: {query3}")
    # print(f"query4: {query4}")
    # print(f"response: {response}")
    # print(f"response_answer: {response_answer}")


    return any("1" in sublist for sublist in [rating1, rating2, rating3, rating4])

if __name__ == '__main__':
    data = {"problem": "2+3=", "subject": "basic", "solution":"2+3=5", "answer": "5", "level": 1, "unique_id": "test/precalculus/807.json"}
    response = get_response(data["problem"])
    # print(response)
    response_answer = ""
    result = get_critic(data["problem"], data["solution"], data["answer"], response, response_answer)
    print(result)