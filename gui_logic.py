from gui import Ui_Dialog

from graph import Graph
from drawer import Drawer as drawer
from data_and_processing import DataAndProcessing

from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')


# Перевод цветной картинки в серую
def black_white_image(color_picture):
    height, width, _ = color_picture.shape

    gray_image = np.zeros((height, width), dtype="uint8")
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = np.uint8(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier

    return gray_image

# Возвращает энергию картинку
def energy_pictures(pictures):
    energy = 0
    for picture_line in pictures:
        for pixel in picture_line:
            energy += pixel * pixel
    return energy


# Считаем критерий разницы изображений
def criterion_difference_images(images1, images2):
    epsilon = 0
    height, width = images1.shape
    for j in range(height):
        for i in range(width):
            epsilon += \
                (images1[j, i] - images2[j, i]) * \
                (images1[j, i] - images2[j, i])
    return epsilon / energy_pictures(images1)


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):
    # Конструктор
    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Кнопка закрытия
            QtCore.Qt.WindowMaximizeButtonHint |  # Кнопка развернуть
            QtCore.Qt.WindowMinimizeButtonHint  # Кнопка свернуть
        )
        # Устанавливаем пользовательский интерфейс
        self.setupUi(dialog)

        # ПОЛЯ КЛАССА
        # Оригинальное изображение
        self.graph_original_picture = Graph(
            layout=self.layout_plot_1,
            widget=self.widget_plot_1,
            name_graphics="Рис. 1. Оригинальное изображение",
            init_toolbar=False
        )
        # Результирующее изображение
        self.graph_resulting_picture = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="Рис. 2. Результирующее изображение",
            init_toolbar=False
        )

        # Картинки обработки
        self.original_picture_color = None
        self.original_picture_gray = None

        self.resulting_picture_gray = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # Смена режима отображения картинки
        self.radioButton_color_picture.clicked.connect(self.display_original_picture)
        self.radioButton_gray_picture.clicked.connect(self.display_original_picture)

        # Алгоритм обработки
        # Загрузка картинки
        self.pushButton_loading_pictures.clicked.connect(self.load_image)
        self.pushButton_processing.clicked.connect(self.processing)

    # ОБРАБОТКА ИНТЕРФЕЙСА
    # отображение оригинальной картинки, с учетом выбора цветового режима
    def display_original_picture(self):
        # Картинки нет - не отображаем
        if self.original_picture_color is None:
            return

        # Проверяем вид отображаемой картинки
        if self.radioButton_color_picture.isChecked():
            drawer.image_color(self.graph_original_picture, self.original_picture_color)
        else:
            drawer.image_gray(self.graph_original_picture, self.original_picture_gray)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (0) Загрузить картинку
    def load_image(self):
        # Вызов окна выбора файла
        # filename, filetype = QFileDialog.getOpenFileName(
        #     None, "Выбрать файл изображения", ".", "All Files(*)")
        filename = "cat.jpg"

        # Загружаем картинку
        self.original_picture_color = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Конвертируем в серый и заносим в класс исходного изображения, рассчитывая гистограммы
        self.original_picture_gray = black_white_image(self.original_picture_color)
        # Отображаем
        self.display_original_picture()

    def processing(self):
        # Картинки нет - сброс
        if self.original_picture_gray is None:
            return

        step_angle = int(self.lineEdit_scan_step.text())

        self.resulting_picture_gray = \
            DataAndProcessing.all_calculation(self.original_picture_gray, step_angle)

        drawer.image_gray(self.graph_resulting_picture, self.resulting_picture_gray)

        # Считаем критерий очистки и отображаем
        epsilon = criterion_difference_images(self.original_picture_gray, self.resulting_picture_gray)
        self.label_deviation_original_and_received.setText(f'{epsilon:.4f}')
