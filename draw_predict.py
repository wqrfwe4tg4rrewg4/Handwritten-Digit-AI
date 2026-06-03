import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# ===== ЗАГРУЗКА ВЕСОВ =====

W1 = np.load("weights_i_h.npy")
W2 = np.load("weights_h_o.npy")
B1 = np.load("bias_i_h.npy")
B2 = np.load("bias_h_o.npy")

# ===== МАТЕМАТИКА =====

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

def predict(img):
    img = img.reshape(784, 1)

    hidden = sigmoid(W1 @ img + B1)
    output = softmax(W2 @ hidden + B2)

    digit = int(np.argmax(output))
    confidence = float(np.max(output))

    return digit, confidence

# ===== ПОДГОТОВКА КАК В MNIST =====

def preprocess(pil_img):

    img = np.array(pil_img.convert("L"))

    coords = np.argwhere(img > 20)

    if len(coords) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    img = img[y0:y1 + 1, x0:x1 + 1]

    img = Image.fromarray(img)
    img = img.resize((20, 20))

    canvas = np.zeros((28, 28), dtype=np.uint8)

    canvas[4:24, 4:24] = np.array(img)

    # ВАЖНО:
    # НЕ ИНВЕРТИРУЕМ ЦВЕТА

    canvas = canvas.astype(np.float32) / 255.0

    return canvas

# ===== ИНТЕРФЕЙС =====

class App:

    def __init__(self, root):

        self.root = root
        self.root.title("Распознавание цифр")

        self.canvas = tk.Canvas(
            root,
            width=280,
            height=280,
            bg="black"
        )
        self.canvas.pack()

        self.result_label = tk.Label(
            root,
            text="Нарисуйте цифру",
            font=("Arial", 16)
        )
        self.result_label.pack()

        tk.Button(
            root,
            text="Распознать",
            command=self.run_prediction
        ).pack(fill="x")

        tk.Button(
            root,
            text="Очистить",
            command=self.clear
        ).pack(fill="x")

        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):

        x = event.x
        y = event.y
        r = 10

        self.canvas.create_oval(
            x-r,
            y-r,
            x+r,
            y+r,
            fill="white",
            outline="white"
        )

        self.draw.ellipse(
            [x-r, y-r, x+r, y+r],
            fill=255
        )

    def clear(self):

        self.canvas.delete("all")

        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(
            text="Нарисуйте цифру"
        )

    def run_prediction(self):

        img = preprocess(self.image)

        Image.fromarray(
            (img * 255).astype(np.uint8)
        ).save("debug.png")

        digit, confidence = predict(img)

        self.result_label.config(
            text=f"Цифра: {digit} ({confidence:.1%})"
        )

        print(
            f"Предсказание: {digit} | Уверенность: {confidence:.1%}"
        )

# ===== ЗАПУСК =====

root = tk.Tk()
app = App(root)
root.mainloop()