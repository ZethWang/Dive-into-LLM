import random
from http import HTTPStatus
from dashscope import Generation

def call_with_messages():
    with open('code\in.txt','r',encoding='utf-8') as f:
      messages = [{'role': 'system', 'content': 'hello'},
                  {'role': 'user', 'content': f.read()}]
      response = Generation.call(model="qwen-turbo",
                                api_key='sk-af00991677444dc5be1c5d06788e03c6',  # 在此处填入你的API Key
                                messages=messages,
                                # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                                seed=random.randint(1, 10000),
                                temperature=0.8,
                                top_p=0.8,
                                top_k=50,
                                # 将输出设置为"message"格式
                                result_format='message')
      if response.status_code == HTTPStatus.OK:
          with open('out.txt', 'w', encoding='utf-8') as file:
              print(response)
              file.write(str(response['output']['choices'][0]['message']['content']))
      else:
          error_message = ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
              response.request_id, response.status_code,
              response.code, response.message
          ))
          with open('code\out.txt', 'w', encoding='utf-8') as file:
              file.write(error_message)

if __name__ == '__main__':
    call_with_messages()
