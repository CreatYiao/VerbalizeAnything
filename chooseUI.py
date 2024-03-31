import tkinter as tk
import main
import Execution
from PIL import Image, ImageTk
from dashscope import MultiModalConversation  # 通义千问
from http import HTTPStatus
import dashscope
dashscope.api_key = 'sk-c8e61bbacecf4348b17f48812d4b5269'

import pyttsx3

img_path = "myImage//"
file_name = ""

Eng = "You are a one of the best English teacher in the world.I am a student who want to study English with you."
Fra = "You are a one of the best French teacher in the world.I am a student who want to study French with you."
Ger = "You are a one of the best German teacher in the world.I am a student who want to study German with you."
Spa = "You are a one of the best Spanish teacher in the world.I am a student who want to study Spanish with you."
Jap = "You are a one of the best Japanese teacher in the world.I am a student who want to study Japanese with you."
Kor = "You are a one of the best Korean teacher in the world.I am a student who want to study Korean with you."

daily = " And the difficulty level of learning is daily conversation."
business = "example sentences should reflect IELTS level"

learn_language = ""
learn_level = ""

lan = ""
dif = ""

class ConfirmationWindow(tk.Tk):  # 如果继承 tk.Toplevel 会开多个窗口
    def __init__(self, language, difficulty):
        super().__init__()

        self.title("选项确认")
        self.language = language
        self.difficulty = difficulty

        tk.Label(self, text="您选择的所学语言是：").pack(anchor=tk.W)
        tk.Label(self, text=language).pack(anchor=tk.W)

        tk.Label(self, text="您所选的学习难度是：").pack(anchor=tk.W)
        tk.Label(self, text=difficulty).pack(anchor=tk.W)

        confirm_button = tk.Button(self, text="Confirm", command=self.open_main_window)
        confirm_button.pack()

    def open_main_window(self):
        self.destroy()
        global lan
        global dif
        lan = self.language
        dif = self.difficulty
        MainWindow()

class LanguageSelection(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("请选择你想学习的语言")
        self.selected_language = tk.StringVar()

        languages = ["英语", "法语", "德语", "西班牙语", "日语", "韩语"]

        tk.Label(self, text="选择语言：").pack(anchor=tk.W)
        for lang in languages:
            tk.Radiobutton(self, text=lang, variable=self.selected_language, value=lang).pack(anchor=tk.W)

        self.next_button = tk.Button(self, text="下一步", command=lambda: self.show_difficulty_selection())
        self.next_button.pack()

    def show_difficulty_selection(self):
        selected_language = self.selected_language.get()
        if not selected_language:
            tk.messagebox.showerror("错误！", "请至少选择一种语言！")
            return

        self.destroy()
        DifficultySelection(selected_language)

class DifficultySelection(tk.Tk):
    def __init__(self, languages):
        super().__init__()

        self.title("请选择你想学习的难度")
        self.selected_difficulty = tk.StringVar()

        difficulties = ["日常会话", "商务交流"]

        tk.Label(self, text="难度选择：").pack(anchor=tk.W)
        for dif in difficulties:
            tk.Radiobutton(self, text=dif, variable=self.selected_difficulty, value=dif).pack(anchor=tk.W)

        self.confirm_button = tk.Button(self, text="确认", command=lambda: self.open_confirmation_window(languages))
        self.confirm_button.pack()

    def open_confirmation_window(self, language):
        selected_difficulty = self.selected_difficulty.get()
        if not selected_difficulty:
            tk.messagebox.showerror("错误！", "请至少选择一种难度！")
            return
        self.destroy()
        ConfirmationWindow(language, selected_difficulty)


def getPromptInfo(language, difficulty):
    global learn_language
    language_info = {
        "英语": "You are a one of the best English teacher in the world. I am a student who want to study English with you.",
        "法语": "You are a one of the best French teacher in the world. I am a student who want to study French with you.",
        "德语": "You are a one of the best German teacher in the world. I am a student who want to study German with you.",
        "西班牙语": "You are a one of the best Spanish teacher in the world. I am a student who want to study Spanish with you.",
        "日语": "You are a one of the best Japanese teacher in the world. I am a student who want to study Japanese with you.",
        "韩语": "You are a one of the best Korean teacher in the world. I am a student who want to study Korean with you."
    }
    difficulty_info = {
        "日常会话": " And the difficulty level of learning is daily conversation.",
        "商务交流": "example sentences should reflect IELTS level"
    }
    # 获取语言对应的提示信息
    learn_language = language_info.get(language, "Unknown Language")

    # 根据难度选择学习水平
    learn_level = difficulty_info.get(difficulty, ", the level is daily talk")

    # 拼接提示信息
    info_text = learn_language + " " + learn_level
    print(info_text)
    return info_text


# 调用通义千问
def call_with_local_file():
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """
    print("filename is: " + file_name)
    print(lan)
    print(dif)
    messages = [{
        'role': 'system',
        'content': [{
            'text': getPromptInfo(lan, dif)
        }]
    }, {
        'role':
            'user',
        'content': [
            {
                'image': "file://" + img_path + file_name + ".jpg"
            },
            {
                'text': 'According to the objects shown in the picture, give the corresponding word, phonetic symbol and relevant example sentence,you dont need to send the picture back. If the object is complex, answer with the closest name.Please return in the following json format {"english_word","phonetic_symbols","relevant_example_sentences"}'
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    # print(response)
    return response


# 打印通义识别的信息
def printInfo():
    response_origin = call_with_local_file()
    response = response_origin
    speak_info = response['output']['choices'][0].message['content'][0]
    print(speak_info)
    read_text(my_speaker(speak_info))
    show_result(speak_info)
    return speak_info


def my_speaker(speak_info):
    # 提取第一个引号和第三个引号的内容
    text_list = speak_info['text'][1:-1].split(',')
    first_quote = text_list[0][1:]
    third_quote = text_list[2][:-1]
    content = first_quote + ',' + third_quote
    return content

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

photo = None

def show_result(result):
    global photo
    # 创建一个顶级窗口
    popup = tk.Tk()
    # 设置窗口标题
    popup.title("识别结果:")
    print(img_path + file_name+".jpg")
    image = Image.open(img_path + file_name + ".jpg")
    # image = image.resize((300, 300))  # 调整图片大小
    photo = ImageTk.PhotoImage(image)

    # 创建标签，显示图片
    label_image = tk.Label(popup, image=photo)
    label_image.image = photo  # 保持图片对象的引用，避免被垃圾回收
    label_image.pack()

    # 创建标签，显示识别结果
    print(type(result['text']))
    label_result = tk.Label(popup, text=result['text'])
    label_result.pack()
    # popup.after(5000, popup.destroy)

    # 运行弹窗
    popup.mainloop()

class MainWindow(tk.Tk):
    def __init__(self):
        # super().__init__()
        # self.title("主程序窗口")
        # global file_name
        self.after(0, self.start_tray_icon())
        # file_name = main.runMySeg()
        # file_name = Execution.cur_file_name
        # printInfo()
        # Your main window content goes here

    def start_tray_icon(self):
        # 执行子线程任务
        start_tray_icon()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import tkinter as tk
from PIL import Image, ImageTk
from pynput import mouse  # 使用 pynput 库监听鼠标点击事件
import pyscreenshot as ImageGrab  # 使用 pyscreenshot 库进行屏幕截取
import threading
from pystray import Icon, Menu, MenuItem
import io
import time
import datetime

# 初始化全局变量
square_regions = []
input_point_camera = None
input_point_screen = None
listener = None  # 鼠标监听器
app_visible = True  # 应用程序窗口是否可见

img_path = "myImage//"
file_name = ""

# 获取当前时间的函数
def getCurTime():
    stamp = int(time.time())
    cur_time = datetime.datetime.fromtimestamp(stamp)
    file_name = cur_time.strftime("%Y%m%d_%H%M%S")
    return file_name

# 鼠标事件回调函数
def on_mouse_camera(event, x, y, flags, param):
    global input_point_camera
    input_point_camera = None
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'你点击了摄像头画面的点：(x={x}, y={y})')
        # 存储点击的坐标
        input_point_camera = np.array([[x, y]])
        # 清空之前的方块区域
        square_regions.clear()

# 启动摄像头
def start_camera():
    global square_regions

    square_regions = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('无法打开摄像头')
        return

    print('摄像头已打开')

    # 定义显示摄像头画面的函数
    def show_camera():
        while True:
            ret, frame = cap.read()
            if not ret:
                print('未能从摄像头读取画面')
                break

            cv2.imshow('Camera', frame)
            cv2.setMouseCallback('Camera', on_mouse_camera)

            key = cv2.waitKey(1) & 0xFF
            if input_point_camera is not None:
                predict_image_region(frame, input_point_camera)
                break

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

    # 在单独的线程中显示摄像头画面
    camera_thread = threading.Thread(target=show_camera)
    camera_thread.start()

    camera_thread.join()

    cap.release()
    cv2.destroyAllWindows()

# 在屏幕上预测图像区域
def predict_image_region(image, input_point):
    global square_regions

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        non_zero_indices = np.nonzero(mask)
        # 获取最上、最下、最左、最右的坐标
        topmost = np.min(non_zero_indices[0])
        bottommost = np.max(non_zero_indices[0])
        leftmost = np.min(non_zero_indices[1])
        rightmost = np.max(non_zero_indices[1])
        top, bottom, left, right = topmost, bottommost, leftmost, rightmost
        square_region = image[top:bottom + 1, left:right + 1]
        square_region_rgb = cv2.cvtColor(square_region, cv2.COLOR_BGR2RGB)
        square_regions.append(square_region_rgb)

        global file_name
        file_name = getCurTime()
        cv2.imwrite(img_path + file_name + f'.jpg', cv2.cvtColor(square_regions[0], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 100])
        # cv2.imwrite(img_path + file_name + f'.jpg', cv2.cvtColor(square_regions[0], cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY, 100])
        printInfo()
        break

        # 显示截取的方块区域
        plt.imshow(square_region_rgb)
        plt.axis('off')
        plt.show()

# 鼠标点击事件回调函数
def on_click(x, y, button, pressed):
    global input_point_screen, listener
    if pressed:
        print(f'你点击了屏幕画面的点：(x={x}, y={y})')
        input_point_screen = np.array([[x, y]])
        # 截图
        screenshot = ImageGrab.grab()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        threading.Thread(target=predict_image_region, args=(screenshot, input_point_screen)).start()
        listener.stop()  # 停止监听

# 开始监听鼠标点击事件
def start_listener():
    global listener
    listener = mouse.Listener(on_click=on_click)
    listener.start()

# 截取屏幕
def take_screenshot():
    start_listener()  # 开始监听

# 创建系统托盘图标
def start_tray_icon():
    icon_path = "icon.png"

    # Read the icon file
    with open(icon_path, "rb") as icon_file:
        icon_data = icon_file.read()

    # Convert the bytes to a PIL Image object
    icon_image = Image.open(io.BytesIO(icon_data))

    # Create the system tray icon
    icon = Icon("Camera Predictor", icon=icon_image, title="Camera Predictor")

    # Tray icon right-click menu
    menu = Menu(
        MenuItem('Start Camera', start_camera),
        MenuItem('Take Screenshot', take_screenshot),
        MenuItem('Exit', lambda: icon.stop()),
    )

    icon.menu = menu
    icon.run()

if __name__ == "__main__":
    app = LanguageSelection()
    app.mainloop()

