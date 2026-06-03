# segment.py — нарезка изображения на отдельные символы
#
# Задача: взять картинку со словом или предложением
# и вернуть список картинок 28×28 — по одной на каждый символ.
#
# Используем OpenCV — библиотеку для обработки изображений.
# Установка: pip install opencv-python

import cv2
import numpy as np


def segment_characters(image_path, debug=False):
    """
    Принимает путь к картинке.
    Возвращает список numpy-массивов формата (28, 28) — по одному на символ.
    Символы отсортированы слева направо.

    debug=True — показывает картинку с нарисованными рамками вокруг букв.
    """

    # Шаг 1: загружаем картинку в оттенках серого
    # Цвет нам не нужен — важна только яркость пикселей
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Картинка не найдена: {image_path}")

    # Шаг 2: бинаризация (пороговая фильтрация)
    # Превращаем серую картинку в чёрно-белую: каждый пиксель либо 0, либо 255.
    # THRESH_BINARY_INV = инвертированная бинаризация:
    #   тёмные пиксели (буквы) → 255 (белые)
    #   светлые пиксели (фон)  → 0   (чёрные)
    # Это нужно потому что OpenCV ищет белые объекты на чёрном фоне.
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Шаг 3: морфологическое замыкание (опционально)
    # Если буква состоит из нескольких кусков (например, "i" = точка + палочка),
    # склеиваем их в один контур
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Шаг 4: поиск контуров
    # Контур — это граница белого объекта на чёрном фоне.
    # RETR_EXTERNAL — берём только внешние контуры (не вложенные)
    # CHAIN_APPROX_SIMPLE — упрощённое представление контура
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Символы не найдены. Проверь картинку.")
        return []

    # Шаг 5: сортируем контуры слева направо по x-координате
    # boundingRect возвращает (x, y, ширина, высота) прямоугольника вокруг контура
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    chars = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Фильтрация мусора: игнорируем слишком маленькие объекты
        # (пыль, артефакты сканирования, точки)
        if w < 5 or h < 5:
            continue

        # Вырезаем прямоугольник с буквой из бинаризованного изображения
        char_img = thresh[y:y+h, x:x+w]

        # Добавляем небольшие отступы вокруг буквы — так она не "упирается" в края
        pad = 4
        char_img = cv2.copyMakeBorder(char_img, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=0)

        # Масштабируем до 28×28 — именно такой формат ждёт наша нейросеть
        char_img = cv2.resize(char_img, (28, 28))

        # Нормализуем: пиксели из [0, 255] в [0, 1]
        char_img = char_img.astype("float32") / 255

        chars.append(char_img)

        # Отладка: рисуем рамку на оригинальной картинке
        if debug:
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if debug:
        cv2.imshow("Сегментация", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Найдено символов: {len(chars)}")
    return chars
