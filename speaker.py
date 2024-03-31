response = call_with_local_file()
response = response['output']['choices'][0].message['content'][0]

# 提取第一个引号和第三个引号的内容
text_list = response['text'][1:-1].split(',')
first_quote = text_list[0][1:]
third_quote = text_list[2][:-1]
content = first_quote+','+third_quote

import pyttsx3

def read_text(text):
    # 初始化语音引擎
    engine = pyttsx3.init()

    # 设置朗读的速度和音量
    engine.setProperty('rate', 150)  # 调整为所需的速度
    engine.setProperty('volume', 1)   # 调整为所需的音量

    # 朗读文本内容
    engine.say(text)

    # 等待语音朗读完成
    engine.runAndWait()

# 测试
text = content
read_text(text)
