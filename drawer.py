import cv2
from graph import Graph
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
# Очистка и подпись графика (вызывается в начале)
def cleaning_and_chart_graph(graph: Graph, x_label, y_label, title):
    if graph.toolbar is not None:
        graph.toolbar.home()  # Возвращаем зум в домашнюю позицию
        graph.toolbar.update()  # Очищаем стек осей (от старых x, y lim)
    # Очищаем график
    graph.axis.clear()
    # Снимаем отображение значений по осям
    graph.axis.get_xaxis().set_visible(False)
    graph.axis.get_yaxis().set_visible(False)
    # Задаем название осей
    graph.axis.set_xlabel(x_label)
    graph.axis.set_ylabel(y_label)
    # Задаем название графика
    graph.axis.set_title(title)


# Отрисовка (вызывается в конце)
def draw_graph(graph: Graph):
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


# Отрисовка при отсутствии данных
def no_data(graph: Graph):
    graph.axis.text(0.5, 0.5, "Нет данных",
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center')
    # Отрисовка, без подписи данных графиков
    draw_graph(graph)


# Класс художник. Имя холст (graph), рисует на нем данные
class Drawer:
    # ПАРАМЕТРЫ ГРАФИКОВ
    horizontal_axis_name_data = "X"
    vertical_axis_name_data = "Y"

    # ОТРИСОВКИ
    # (1) Цветная картинка
    @staticmethod
    def image_color(
            graph: Graph,
            data
    ):
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=Drawer.horizontal_axis_name_data, y_label=Drawer.vertical_axis_name_data
        )

        # Рисуем график
        im = graph.axis.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        # Если color bar нет- создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (2) Серая картинка
    @staticmethod
    def image_gray(
            graph: Graph,
            data
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=Drawer.horizontal_axis_name_data, y_label=Drawer.vertical_axis_name_data
        )

        # Рисуем график
        im = graph.axis.imshow(data, cmap='gray',vmin=0, vmax=255 )
        # Если color bar нет- создаем, иначе обновляем
        if not graph.colorbar:
            divider = make_axes_locatable(graph.axis)
            cax = divider.append_axes("right", "10%", pad="3%")
            graph.colorbar = graph.figure.colorbar(im, orientation='vertical', cax=cax)
        else:
            graph.colorbar.update_normal(im)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    # (3) Гистограмма
    @staticmethod
    def histogram(
            graph: Graph,
            data
    ):

        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=Drawer.horizontal_axis_name_data, y_label=Drawer.vertical_axis_name_data
        )

        # Рисуем график
        graph.axis.bar(data.index, height=data.values, width=1)

        # Отрисовка (вызывается в конце)
        draw_graph(graph)