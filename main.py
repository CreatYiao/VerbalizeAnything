import tkinter as tk
import cv2
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, master):
        self.master = master
        master.title("Camera App")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(master, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(master, text="Take Snapshot", width=20, command=self.snapshot)
        self.btn_snapshot.pack(pady=10)

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 转换颜色通道顺序
            frame_rgb2 =frame_rgb[:,:,::-1]
            cv2.imwrite("snapshot.png", frame_rgb2)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 转换颜色通道顺序
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

def main():
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()