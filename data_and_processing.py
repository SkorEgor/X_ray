import numpy as np
import math
import kaczmarz


class DataAndProcessing:
    # Конструктор
    def __init__(self):
        pass

    # СКАНИРОВАНИЕ
    # справа или слева
    @staticmethod
    def scanning_right_or_left(matrix, angle_degrees, left=True):
        # Угол пролета лучей
        angle_radians = angle_degrees * math.pi / 180.
        k = math.tan(angle_radians)

        # Параметры матрицы
        height, width = matrix.shape
        number_pixels = height * width

        # Режим сканирования: слева или справа
        if left:
            x_0 = 0
            range_x = range(width)

        else:
            x_0 = width - 1
            range_x = range(width - 1, -1, -1)

        # Матрица результатов
        used_indexes = np.zeros(
            (height, number_pixels),
            dtype=bool)
        resulting_sum = np.zeros(
            height,
            dtype=int)
        record_line = 0

        # Проходим по боковой стороне
        for y_0 in range(height):
            # Пропуская луч
            for x in range_x:
                # КООРДИНАТЫ ЛУЧА
                y = k * (x - x_0) + y_0

                # ПРОЙДЕННЫЕ ИНДЕКСЫ
                i = round(x)
                j = round(y)

                # Выход луча за пределы
                if not (0 <= j <= height - 1):
                    break

                # ЗАПИСЬ РЕЗУЛЬТАТОВ
                # запоминаем индекс пикселя через который прошел луч
                used_indexes[record_line][j * width + i] = True
                # Добавляем его значение к результирующей сумме
                resulting_sum[record_line] += matrix[j][i]

            # Переводим на следующую строку записи, для следующего луча
            record_line += 1

        # Матрица использованных индексов и их суммы
        return used_indexes, resulting_sum

    # сверху или снизу
    @staticmethod
    def scan_bottom_or_top(matrix, angle_degrees, bottom=True):
        # Угол пролета лучей
        angle_radians = angle_degrees * math.pi / 180.
        k = math.tan(angle_radians)

        # Параметры матрицы
        height, width = matrix.shape
        number_pixels = height * width

        # Режим сканирования: снизу или сверху
        y_0 = 0

        if bottom:
            y_0 = 0
            range_y = range(height)

        else:
            y_0 = height - 1
            range_y = range(height - 1, -1, -1)

        # Матрица результатов
        used_indexes = np.zeros(
            (width, number_pixels),
            dtype=bool)
        resulting_sum = np.zeros(
            width,
            dtype=int)
        record_line = 0

        # Проходим по верхней или нижней стороне
        for x_0 in range(width):

            # Пропуская луч
            for y in range_y:
                # КООРДИНАТЫ ЛУЧА
                x = (y - y_0) / k + x_0

                # ПРОЙДЕННЫЕ ИНДЕКСЫ
                i = round(x)
                j = round(y)

                # Выход луча за пределы
                if not (0 <= i <= width - 1):
                    break

                # ЗАПИСЬ РЕЗУЛЬТАТОВ
                # запоминаем индекс пикселя через который прошел луч
                used_indexes[record_line][j * width + i] = True
                # Добавляем его значение к результирующей сумме
                resulting_sum[record_line] += matrix[j][i]

            # Переводим на следующую строку записи, для следующего луча
            record_line += 1

        # Матрица использованных индексов и их суммы
        return used_indexes, resulting_sum

    # ПРОЦЕСС ВОСТАНОВЛЕНИЯ.
    # Метод 1) Наименьших квадратов / Псевдообратной матрицы
    @staticmethod
    def method_least_squares(a, b, c=None):
        a_reverse = np.linalg.pinv(a)
        return np.matmul(a_reverse, b)

    # Метод 2) Качмарж с правилом циклического выбора
    @staticmethod
    def method_karchmarg_cyclical(a, b, max_iter=5000):
        a_resulting = kaczmarz.Cyclic.solve(a, b, maxiter=max_iter)
        return a_resulting

    # Метод 3) Качмарж с правилом максимального расстояния
    @staticmethod
    def method_karchmarg_max_distance(a, b, max_iter=5000):
        a_resulting = kaczmarz.MaxDistance.solve(a, b, maxiter=max_iter)
        return a_resulting

    # ***) Возвращает не повторяющие строки
    @staticmethod
    def no_repetitions(matrix_a, vector_b):
        matrix_for_calculation = [tuple(row) for row in matrix_a]
        matrix_a_without_repetitions, indexes_of_non_repeating_rows = np.unique(
            matrix_for_calculation, axis=0, return_index=True)
        return matrix_a_without_repetitions, vector_b[indexes_of_non_repeating_rows]

    # ОБЩАЯ ФУНКЦИЯ
    @staticmethod
    def full_scan(picture_gray, multiple_angle=15):
        # ЗАДЕМ НАЧАЛЬНОЕ СКАНИРОВАНИЕ
        first_scan_angle = 90
        matrix_a, vector_b = DataAndProcessing.scan_bottom_or_top(picture_gray, first_scan_angle, bottom=True)

        # ДОПОЛНИТЕЛЬНОЕ СКАНИРОВАНИЕ
        # 1. Сканируем СНИЗУ 0 - 180
        for scan_angle in range(multiple_angle, 180, multiple_angle):
            if scan_angle == first_scan_angle:
                continue

            matrix_a_new, vector_b_new = DataAndProcessing.scan_bottom_or_top(
                picture_gray, scan_angle, bottom=True)

            matrix_a = np.vstack([matrix_a, matrix_a_new])
            vector_b = np.concatenate([vector_b, vector_b_new])

        # 2. Сканируем СВЕРХУ 180-360
        for scan_angle in range(180 + multiple_angle, 360, multiple_angle):
            if scan_angle == 90:
                continue

            matrix_a_new, vector_b_new = DataAndProcessing.scan_bottom_or_top(
                picture_gray, scan_angle, bottom=False)

            matrix_a = np.vstack([matrix_a, matrix_a_new])
            vector_b = np.concatenate([vector_b, vector_b_new])

        # 3. Сканируем СЛЕВА 0-90 и 270-360
        for scan_angle in range(-180 + multiple_angle, 90, multiple_angle):
            matrix_a_new, vector_b_new = DataAndProcessing.scanning_right_or_left(
                picture_gray, scan_angle, left=True)

            matrix_a = np.vstack([matrix_a, matrix_a_new])
            vector_b = np.concatenate([vector_b, vector_b_new])

        # # 4. Сканируем Справа 90-270
        for scan_angle in range(90 + multiple_angle, 270, multiple_angle):
            matrix_a_new, vector_b_new = DataAndProcessing.scanning_right_or_left(
                picture_gray, scan_angle, left=True)

            matrix_a = np.vstack([matrix_a, matrix_a_new])
            vector_b = np.concatenate([vector_b, vector_b_new])

        return matrix_a, vector_b

    @staticmethod
    def recovery_by_scans(matrix_a, vector_b, shape,
                          solution_method, max_iter=5000):
        # ПАРАМЕТРЫ ИЗОБРАЖЕНИЯ
        height, width = shape[0], shape[1]
        # matrix_a = matrix_a.astype(float)

        # rez = calculation_pixel_value(matrix_a.astype('float'),b.astype('float')).reshape(height, width).astype('int')
        rez = solution_method(matrix_a, vector_b, max_iter)
        # Приводим к размерам изображения и соответствующему типу данных
        rez = rez.reshape(height, width).astype('int')

        return rez

    @staticmethod
    def all_calculation(picture_gray, solution_method, multiple_angle=15, max_iter=5000):
        # Матрица активных индексов и вектор суммы по ним
        matrix_a, vector_b = DataAndProcessing.full_scan(picture_gray, multiple_angle)
        # Удаление повторяющихся строк
        matrix_a, vector_b = DataAndProcessing.no_repetitions(matrix_a, vector_b)
        # Находим изображение с заданным размером, через псевдообратную матрицу
        print("g")
        reconstructed_image = DataAndProcessing.recovery_by_scans(matrix_a, vector_b, picture_gray.shape,
                                                                  solution_method, max_iter)
        # Возвращаем результат
        return reconstructed_image
