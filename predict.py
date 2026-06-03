# predict.py — распознавание текста на картинке
#
# Этот файл объединяет всё:
#   1. Загружает обученные веса из файлов .npy
#   2. Нарезает картинку на символы (segment.py)
#   3. Прогоняет каждый символ через нейросеть
#   4. Склеивает результат в строку
#
# Запускай ТОЛЬКО после того как запустил main.py и веса сохранились!

import numpy as np
from segment import segment_characters
from utils import CLASSES


# ── Загрузка весов ────────────────────────────────────────────────────────────
# Загружаем веса которые нейросеть "выучила" во время обучения в main.py
try:
    weights_i_h = np.load("weights_i_h.npy")
    weights_h_o = np.load("weights_h_o.npy")
    bias_i_h    = np.load("bias_i_h.npy")
    bias_h_o    = np.load("bias_h_o.npy")
    print("Веса загружены успешно")
except FileNotFoundError:
    print("Ошибка: файлы весов не найдены.")
    print("Сначала запусти main.py чтобы обучить сеть и сохранить веса.")
    exit(1)


# ── Функция forward pass ──────────────────────────────────────────────────────
def predict_char(image_28x28):
    """
    Принимает одно изображение символа (28×28).
    Возвращает предсказанный символ и уверенность (0–100%).
    """
    # Превращаем матрицу 28×28 в столбец 784×1
    image = image_28x28.reshape(-1, 1)

    # Прямое распространение — точно как в main.py
    hidden = 1 / (1 + np.exp(-(bias_i_h + weights_i_h @ image)))
    output = 1 / (1 + np.exp(-(bias_h_o + weights_h_o @ hidden)))

    # argmax — индекс нейрона с максимальным значением = предсказанный класс
    predicted_index = output.argmax()
    confidence = float(output[predicted_index]) * 100

    return CLASSES[predicted_index], confidence


# ── Функция распознавания всего текста ───────────────────────────────────────
def predict_text(image_path, show_details=True):
    """
    Принимает путь к картинке со словом или предложением.
    Возвращает распознанный текст.

    show_details=True — печатает каждый символ с уверенностью.
    """
    # Нарезаем картинку на символы
    chars = segment_characters(image_path)

    if not chars:
        return ""

    result = ""

    for i, char_img in enumerate(chars):
        char, confidence = predict_char(char_img)
        result += char

        if show_details:
            bar = "█" * int(confidence / 10) + "░" * (10 - int(confidence / 10))
            print(f"  Символ {i+1}: '{char}'  {bar}  {confidence:.1f}%")

    return result


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Можно передать путь к картинке аргументом: python predict.py моя_картинка.jpg
    # Или редактируй строку ниже
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_word.jpg"  # ← замени на свою картинку

    print(f"\nРаспознаю: {image_path}")
    print("─" * 40)

    text = predict_text(image_path, show_details=True)

    print("─" * 40)
    print(f"Результат: {text}")
