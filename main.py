import numpy as np
from tensorflow import keras
import utils

# ── Загрузка данных ─────────────────────────────────────────────
print("Загружаю MNIST...")

images, labels = utils.load_dataset()

print(f"Загружено {images.shape[0]} изображений")

# ── Гиперпараметры ──────────────────────────────────────────────
input_size = 784
hidden_size = 64
output_size = 10   # ❗️ MNIST = 10 классов

epochs = 5
learning_rate = 0.05

# ── Инициализация весов ─────────────────────────────────────────
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))

bias_input_to_hidden = np.zeros((hidden_size, 1))
bias_hidden_to_output = np.zeros((output_size, 1))

# ── Softmax ──────────────────────────────────────────────────────
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=0, keepdims=True)

# ── Обучение ────────────────────────────────────────────────────
for epoch in range(epochs):
    e_loss = 0
    e_correct = 0

    for image, label in zip(images, labels):

        # reshape
        image = image.reshape(-1, 1)
        label = label.reshape(-1, 1)

        # ── forward ──
        hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden
        hidden = 1 / (1 + np.exp(-hidden_raw))  # sigmoid

        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        output = softmax(output_raw)

        # ── loss (cross-entropy) ──
        e_loss += -np.sum(label * np.log(output + 1e-9))
        e_correct += int(np.argmax(output) == np.argmax(label))

        # ── backprop ──
        delta_output = output - label

        weights_hidden_to_output += -learning_rate * delta_output @ hidden.T
        bias_hidden_to_output += -learning_rate * delta_output

        delta_hidden = (weights_hidden_to_output.T @ delta_output) * (hidden * (1 - hidden))

        weights_input_to_hidden += -learning_rate * delta_hidden @ image.T
        bias_input_to_hidden += -learning_rate * delta_hidden

    print(
        f"Эпоха {epoch+1}/{epochs} | "
        f"Loss: {round(e_loss/len(images), 4)} | "
        f"Accuracy: {round(e_correct/len(images)*100, 2)}%"
    )

# ── Сохранение ────────────────────────────────────────────────
np.save("weights_i_h.npy", weights_input_to_hidden)
np.save("weights_h_o.npy", weights_hidden_to_output)
np.save("bias_i_h.npy", bias_input_to_hidden)
np.save("bias_h_o.npy", bias_hidden_to_output)

print("Готово! Теперь запускай predict.py")