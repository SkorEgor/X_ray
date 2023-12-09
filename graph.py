from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


# Класс для объектов графика
class Graph:
    def __init__(self, layout, widget, layout_toolbar=None, name_graphics=None, init_toolbar=False):
        # Объекты графика
        self.axis = None
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.layout = layout  # Слой - для отрисовки графика
        self.widget = widget  # Виджет - для отрисовки графика
        self.colorbar = None
        self.name_graphics = name_graphics
        # Если передали отдельный слой для toolbar, помещаем его туда
        if layout_toolbar is None:
            self.layout_toolbar = layout
        else:
            self.layout_toolbar = layout_toolbar
        # Вызываем инициализацию
        self.initialize(init_toolbar=init_toolbar)

    def initialize(self, draw=False, init_toolbar=False):
        # Инициализирует фигуру matplotlib внутри контейнера GUI.
        # Вызываем только один раз при инициализации

        # Создание фигуры (self.fig и self.ax)
        self.figure = Figure()
        self.axis = self.figure.add_subplot(111)
        # Создание холста
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        if draw:
            self.canvas.draw()

        # Создание Toolbar
        if init_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self.widget, coordinates=True)
            self.layout_toolbar.addWidget(self.toolbar)

    # Приближает указанную область
    def zoom_area(self, x_min, x_max, y_min, y_max):
        # На графике задаем область
        self.toolbar.push_current()  # Сохранить текущий статус zoom как домашний

        self.axis.set_xlim([x_min, x_max])
        self.axis.set_ylim([y_min, y_max])

        # Перерисовываем
        self.canvas.draw()
