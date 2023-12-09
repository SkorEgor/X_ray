# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1187, 498)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_design = QtWidgets.QWidget(Dialog)
        self.widget_design.setStyleSheet("\n"
"   QWidget{\n"
"    background-color: rgb(33, 37, 43);\n"
"    color: rgb(208, 208, 208);\n"
"    font-size: 10pt;\n"
"}\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
" QScrollBar {\n"
"border: none;                                                /* без границ */\n"
"    border-right:5px solid rgb(211, 211, 211);;    /* С правой красной раницей */\n"
" }\n"
" QScrollBar:vertical {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 8px;\n"
"    margin: 21px 0 21px 0;\n"
"    border-radius: 0px;\n"
" }\n"
"\n"
"/* Ползунок */\n"
" QScrollBar::handle:vertical {    \n"
"    background:rgb(255, 255, 255);\n"
"    min-height: 25px;\n"
"    border-radius: 4px\n"
" }\n"
"/*Нижняя стрелка*/\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
"/*Верхняя стрелка*/\n"
" QScrollBar::sub-line:vertical {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
"/* Цвета нижних и верхних стрелок */\n"
" QScrollBar::up-arrow:vertical{\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     background: rgb(255, 255, 255);\n"
" }\n"
"QScrollBar::down-arrow:vertical{\n"
"    border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     background: rgb(255, 255, 255);\n"
"}\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"RadioButton */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"    width: 15px;\n"
"    height: 15px;\n"
"    border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid rgb(255, 255, 255);\n"
"    border: 3px solid rgb(52, 59, 72);    \n"
"}\n"
"/* срабатывает, когда пользователь наводит на элемент мышью */\n"
"QRadioButton:hover {\n"
"    background-color:rgb(40, 44, 52);            /* задаем цвет фона */\n"
"}\n"
"/* срабатывает, при нажатии*/\n"
"QRadioButton:pressed      {\n"
"    background-color:rgb(170, 170, 170);        /* задаем цвет фона */\n"
"    color:  rgb(0, 0, 0);\n"
"    border: none;                                                /* без границ */\n"
"}\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QCheckBox */\n"
"/* Стандартное состояние*/\n"
"QCheckBox{\n"
"    padding-left: 8px;        /* Отступ слева */\n"
"    padding-right: -8px;    /* Отступ справа */\n"
"}\n"
"\n"
"/* Состояние - не выбран*/\n"
"QCheckBox::indicator:unchecked {\n"
"    /* Выбор картинки*/\n"
"    image: url(:/checkbox_status_success/resource/checkbox_status_success/check_error_red_24dp.svg)\n"
"}\n"
"\n"
"/* Состояние -  выбран*/\n"
"QCheckBox::indicator:checked {\n"
"    /* Выбор картинки*/\n"
"    image: url(:/checkbox_status_success/resource/checkbox_status_success/check_ok_grean_24dp.svg);\n"
"}\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QPushButton */\n"
"/*Стандартное состояние для кнопки*/\n"
"QPushButton {\n"
"    font-size: 12pt;\n"
"    background-color:rgb(37, 41, 48);/* задает цвет фона */\n"
"    display: inline-block;                            /* пределяет, будет ли элемент обрабатываться как блочный или встроенный элемент */\n"
"    border: 1px solid rgb(52, 59, 72);        /* задает границу элемента */\n"
"\n"
"    /* задает иконку */\n"
"    background-position: left center;                            /* выравнивание иконки */\n"
"    background-repeat: no-repeat;                                /* повторять иконку */\n"
"} \n"
"\n"
"/* срабатывает, когда пользователь наводит на элемент мышью */\n"
"QPushButton:hover {\n"
"    background-color:rgb(40, 44, 52);            /* задаем цвет фона */\n"
"    border: none;                                                /* без границ */\n"
"    border-left:4px solid rgb(208, 208, 208);    /* С правой красной раницей */\n"
"}\n"
"\n"
"\n"
"/* срабатывает, при нажатии*/\n"
"QPushButton:pressed      {\n"
"    background-color:rgb(170, 170, 170);        /* задаем цвет фона */\n"
"    color: rgb(181, 181, 181);\n"
"    border: none;                                                /* без границ */\n"
"}\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QLineEdit */\n"
"/* Стиль по умолчанию */\n"
"QLineEdit:enabled{\n"
"    background-color:rgb(44, 49, 58); /* Устанавливаем цвет заливки */\n"
"    border: 1px solid rgb(255, 255, 255); \n"
"}\n"
"\n"
"/* Если поле отключено */\n"
"QLineEdit:disabled {\n"
"    background-color:  rgba(67, 74, 88, 0); /* Устанавливаем цвет заливки */\n"
"    border: 1px solid rgb(255, 255, 255); \n"
"    color: rgb(67, 74, 88);\n"
"}\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QGroupBox */\n"
"QGroupBox{\n"
"    color:rgb(255, 255, 255);    /* задает цвет шрифта */\n"
"}\n"
"/* ///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTableWidget */\n"
"\n"
"QTableWidget {    \n"
"    gridline-color: rgb(136, 136, 136);\n"
"    border-top: 1px solid rgb(54, 60, 74);\n"
"    border-bottom: 1px solid  rgb(54, 60, 74);\n"
"}\n"
"QTableWidget::item:selected{\n"
"    background-color: rgb(72, 81, 94);\n"
"}\n"
"QHeaderView { qproperty-defaultAlignment: AlignCenter; }\n"
"/*Цвет верхнего и левого поля*/\n"
"QHeaderView::section{\n"
"    background-color:rgb(37, 41, 48);\n"
"    border-style: none;\n"
"border: 1px solid rgb(136, 136, 136);\n"
"}\n"
"/*Кнопка в верхнем левом углу*/\n"
"QTableCornerButton::section {background-color:rgb(33, 37, 43); }\n"
"\n"
"QWidget#widget_loading_data {\n"
"    border-bottom: 2px solid rgb(208, 208, 208);\n"
"}\n"
"QWidget#widget_processing {\n"
"    border-top: 2px solid rgb(208, 208, 208);\n"
"}\n"
"QScrollArea{\n"
"border: none;\n"
"}")
        self.widget_design.setObjectName("widget_design")
        self.layout_design = QtWidgets.QHBoxLayout(self.widget_design)
        self.layout_design.setContentsMargins(0, 0, 0, 0)
        self.layout_design.setSpacing(3)
        self.layout_design.setObjectName("layout_design")
        self.widget_menu = QtWidgets.QWidget(self.widget_design)
        self.widget_menu.setMinimumSize(QtCore.QSize(200, 0))
        self.widget_menu.setMaximumSize(QtCore.QSize(200, 16777215))
        self.widget_menu.setObjectName("widget_menu")
        self.layout_menu = QtWidgets.QVBoxLayout(self.widget_menu)
        self.layout_menu.setContentsMargins(0, 0, 0, 6)
        self.layout_menu.setSpacing(6)
        self.layout_menu.setObjectName("layout_menu")
        self.scrollArea = QtWidgets.QScrollArea(self.widget_menu)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 200, 443))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setContentsMargins(0, -1, 3, 0)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_loading_data = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_loading_data.setMinimumSize(QtCore.QSize(0, 30))
        self.pushButton_loading_data.setCheckable(True)
        self.pushButton_loading_data.setChecked(True)
        self.pushButton_loading_data.setObjectName("pushButton_loading_data")
        self.verticalLayout_2.addWidget(self.pushButton_loading_data)
        self.widget_loading_data = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget_loading_data.setStyleSheet("")
        self.widget_loading_data.setObjectName("widget_loading_data")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_loading_data)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 6)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_loading_pictures = QtWidgets.QWidget(self.widget_loading_data)
        self.widget_loading_pictures.setEnabled(True)
        self.widget_loading_pictures.setStyleSheet("")
        self.widget_loading_pictures.setObjectName("widget_loading_pictures")
        self.layout_loading_pictures = QtWidgets.QVBoxLayout(self.widget_loading_pictures)
        self.layout_loading_pictures.setContentsMargins(0, 0, 0, 5)
        self.layout_loading_pictures.setSpacing(0)
        self.layout_loading_pictures.setObjectName("layout_loading_pictures")
        self.widget_path_to_picture = QtWidgets.QWidget(self.widget_loading_pictures)
        self.widget_path_to_picture.setObjectName("widget_path_to_picture")
        self.layout_path_to_picture = QtWidgets.QHBoxLayout(self.widget_path_to_picture)
        self.layout_path_to_picture.setContentsMargins(9, 0, 9, 2)
        self.layout_path_to_picture.setObjectName("layout_path_to_picture")
        self.label_text_loading_pictures = QtWidgets.QLabel(self.widget_path_to_picture)
        self.label_text_loading_pictures.setObjectName("label_text_loading_pictures")
        self.layout_path_to_picture.addWidget(self.label_text_loading_pictures)
        self.pushButton_loading_pictures = QtWidgets.QPushButton(self.widget_path_to_picture)
        self.pushButton_loading_pictures.setMinimumSize(QtCore.QSize(35, 30))
        self.pushButton_loading_pictures.setMaximumSize(QtCore.QSize(35, 30))
        self.pushButton_loading_pictures.setStyleSheet("/* срабатывает, когда пользователь наводит на элемент мышью */\n"
"QPushButton:hover {\n"
"    border: none;                                                /* без границ */\n"
"}")
        self.pushButton_loading_pictures.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/menu/resource/file_download_white_24dp.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_loading_pictures.setIcon(icon)
        self.pushButton_loading_pictures.setIconSize(QtCore.QSize(25, 25))
        self.pushButton_loading_pictures.setObjectName("pushButton_loading_pictures")
        self.layout_path_to_picture.addWidget(self.pushButton_loading_pictures)
        self.layout_loading_pictures.addWidget(self.widget_path_to_picture)
        self.groupBox_picture_type = QtWidgets.QGroupBox(self.widget_loading_pictures)
        self.groupBox_picture_type.setEnabled(True)
        self.groupBox_picture_type.setObjectName("groupBox_picture_type")
        self.layout_picture_type = QtWidgets.QHBoxLayout(self.groupBox_picture_type)
        self.layout_picture_type.setContentsMargins(9, 5, 0, 5)
        self.layout_picture_type.setObjectName("layout_picture_type")
        self.radioButton_color_picture = QtWidgets.QRadioButton(self.groupBox_picture_type)
        self.radioButton_color_picture.setEnabled(True)
        self.radioButton_color_picture.setCheckable(True)
        self.radioButton_color_picture.setChecked(False)
        self.radioButton_color_picture.setObjectName("radioButton_color_picture")
        self.layout_picture_type.addWidget(self.radioButton_color_picture)
        self.radioButton_gray_picture = QtWidgets.QRadioButton(self.groupBox_picture_type)
        self.radioButton_gray_picture.setChecked(True)
        self.radioButton_gray_picture.setObjectName("radioButton_gray_picture")
        self.layout_picture_type.addWidget(self.radioButton_gray_picture)
        self.layout_loading_pictures.addWidget(self.groupBox_picture_type)
        self.verticalLayout_3.addWidget(self.widget_loading_pictures)
        self.verticalLayout_2.addWidget(self.widget_loading_data)
        self.pushButton_methods = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_methods.setMinimumSize(QtCore.QSize(0, 30))
        self.pushButton_methods.setCheckable(True)
        self.pushButton_methods.setChecked(True)
        self.pushButton_methods.setObjectName("pushButton_methods")
        self.verticalLayout_2.addWidget(self.pushButton_methods)
        self.widget_options = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget_options.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_options.setObjectName("widget_options")
        self.layout_options = QtWidgets.QVBoxLayout(self.widget_options)
        self.layout_options.setContentsMargins(0, 0, 0, 20)
        self.layout_options.setObjectName("layout_options")
        self.groupBox_stretch = QtWidgets.QGroupBox(self.widget_options)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_stretch.sizePolicy().hasHeightForWidth())
        self.groupBox_stretch.setSizePolicy(sizePolicy)
        self.groupBox_stretch.setObjectName("groupBox_stretch")
        self.layout_stretch = QtWidgets.QVBoxLayout(self.groupBox_stretch)
        self.layout_stretch.setContentsMargins(9, 5, 9, 9)
        self.layout_stretch.setSpacing(6)
        self.layout_stretch.setObjectName("layout_stretch")
        self.pushButton_stretch = QtWidgets.QPushButton(self.groupBox_stretch)
        self.pushButton_stretch.setStyleSheet("\n"
"QPushButton {\n"
"    font-size: 10pt;\n"
"    background-color:rgb(37, 41, 48);/* задает цвет фона */\n"
"    display: inline-block;                            /* пределяет, будет ли элемент обрабатываться как блочный или встроенный элемент */\n"
"    border: 1px solid rgb(52, 59, 72);        /* задает границу элемента */\n"
"\n"
"    /* задает иконку */\n"
"    background-position: left center;                            /* выравнивание иконки */\n"
"    background-repeat: no-repeat;                                /* повторять иконку */\n"
"} \n"
"\n"
"/* срабатывает, когда пользователь наводит на элемент мышью */\n"
"QPushButton:hover {\n"
"    background-color:rgb(40, 44, 52);            /* задаем цвет фона */\n"
"    border: none;                                                /* без границ */\n"
"}\n"
"\n"
"\n"
"/* срабатывает, при нажатии*/\n"
"QPushButton:pressed      {\n"
"    background-color:rgb(170, 170, 170);        /* задаем цвет фона */\n"
"    color: rgb(181, 181, 181);\n"
"    border: none;                                                /* без границ */\n"
"}\n"
"/*")
        self.pushButton_stretch.setObjectName("pushButton_stretch")
        self.layout_stretch.addWidget(self.pushButton_stretch)
        self.layout_options.addWidget(self.groupBox_stretch)
        self.widget_normalization = QtWidgets.QWidget(self.widget_options)
        self.widget_normalization.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_normalization.setObjectName("widget_normalization")
        self.layout_normalization = QtWidgets.QVBoxLayout(self.widget_normalization)
        self.layout_normalization.setContentsMargins(0, 0, 0, 20)
        self.layout_normalization.setObjectName("layout_normalization")
        self.groupBox_scan_step = QtWidgets.QGroupBox(self.widget_normalization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_scan_step.sizePolicy().hasHeightForWidth())
        self.groupBox_scan_step.setSizePolicy(sizePolicy)
        self.groupBox_scan_step.setObjectName("groupBox_scan_step")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_scan_step)
        self.verticalLayout.setContentsMargins(9, 5, 9, 9)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_parameters_step_by_angle = QtWidgets.QWidget(self.groupBox_scan_step)
        self.widget_parameters_step_by_angle.setObjectName("widget_parameters_step_by_angle")
        self.layout_parameters_step_by_angle = QtWidgets.QHBoxLayout(self.widget_parameters_step_by_angle)
        self.layout_parameters_step_by_angle.setContentsMargins(0, 0, 0, 0)
        self.layout_parameters_step_by_angle.setObjectName("layout_parameters_step_by_angle")
        self.label_text_scan_step = QtWidgets.QLabel(self.widget_parameters_step_by_angle)
        self.label_text_scan_step.setObjectName("label_text_scan_step")
        self.layout_parameters_step_by_angle.addWidget(self.label_text_scan_step)
        self.lineEdit_scan_step = QtWidgets.QLineEdit(self.widget_parameters_step_by_angle)
        self.lineEdit_scan_step.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_scan_step.setObjectName("lineEdit_scan_step")
        self.layout_parameters_step_by_angle.addWidget(self.lineEdit_scan_step)
        self.verticalLayout.addWidget(self.widget_parameters_step_by_angle)
        self.label_text_note_scan = QtWidgets.QLabel(self.groupBox_scan_step)
        self.label_text_note_scan.setTextFormat(QtCore.Qt.AutoText)
        self.label_text_note_scan.setScaledContents(False)
        self.label_text_note_scan.setWordWrap(True)
        self.label_text_note_scan.setObjectName("label_text_note_scan")
        self.verticalLayout.addWidget(self.label_text_note_scan)
        self.layout_normalization.addWidget(self.groupBox_scan_step)
        self.layout_options.addWidget(self.widget_normalization)
        self.verticalLayout_2.addWidget(self.widget_options)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout_menu.addWidget(self.scrollArea)
        self.pushButton_processing = QtWidgets.QPushButton(self.widget_menu)
        self.pushButton_processing.setObjectName("pushButton_processing")
        self.layout_menu.addWidget(self.pushButton_processing)
        self.widget = QtWidgets.QWidget(self.widget_menu)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_text_deviation_original_and_received = QtWidgets.QLabel(self.widget)
        self.label_text_deviation_original_and_received.setObjectName("label_text_deviation_original_and_received")
        self.horizontalLayout_2.addWidget(self.label_text_deviation_original_and_received)
        self.label_deviation_original_and_received = QtWidgets.QLabel(self.widget)
        self.label_deviation_original_and_received.setText("")
        self.label_deviation_original_and_received.setObjectName("label_deviation_original_and_received")
        self.horizontalLayout_2.addWidget(self.label_deviation_original_and_received)
        self.layout_menu.addWidget(self.widget)
        self.layout_design.addWidget(self.widget_menu)
        self.widget_main = QtWidgets.QWidget(self.widget_design)
        self.widget_main.setStyleSheet("QWidget{\n"
"background-color: rgb(240, 240, 240);\n"
"color: rgb(0, 0, 0);\n"
"}")
        self.widget_main.setObjectName("widget_main")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_main)
        self.gridLayout.setContentsMargins(9, -1, -1, -1)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_plot_1 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_1.setObjectName("widget_plot_1")
        self.layout_plot_1 = QtWidgets.QVBoxLayout(self.widget_plot_1)
        self.layout_plot_1.setContentsMargins(0, 0, 0, 0)
        self.layout_plot_1.setSpacing(0)
        self.layout_plot_1.setObjectName("layout_plot_1")
        self.gridLayout.addWidget(self.widget_plot_1, 0, 0, 1, 1)
        self.widget_plot_2 = QtWidgets.QWidget(self.widget_main)
        self.widget_plot_2.setObjectName("widget_plot_2")
        self.layout_plot_2 = QtWidgets.QVBoxLayout(self.widget_plot_2)
        self.layout_plot_2.setContentsMargins(0, 0, 0, 0)
        self.layout_plot_2.setSpacing(0)
        self.layout_plot_2.setObjectName("layout_plot_2")
        self.gridLayout.addWidget(self.widget_plot_2, 0, 1, 1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.layout_design.addWidget(self.widget_main)
        self.horizontalLayout.addWidget(self.widget_design)

        self.retranslateUi(Dialog)
        self.pushButton_methods.clicked['bool'].connect(self.widget_options.setVisible) # type: ignore
        self.pushButton_loading_data.clicked['bool'].connect(self.widget_loading_data.setVisible) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Фильтрация изображения"))
        self.pushButton_loading_data.setText(_translate("Dialog", "Данные"))
        self.label_text_loading_pictures.setText(_translate("Dialog", "Загрузка файла"))
        self.groupBox_picture_type.setTitle(_translate("Dialog", "Вид отображения"))
        self.radioButton_color_picture.setText(_translate("Dialog", "Цветная"))
        self.radioButton_gray_picture.setText(_translate("Dialog", "Серая"))
        self.pushButton_methods.setText(_translate("Dialog", "Методы"))
        self.groupBox_stretch.setTitle(_translate("Dialog", "Растяжка гистограммы"))
        self.pushButton_stretch.setText(_translate("Dialog", "Обработка"))
        self.groupBox_scan_step.setTitle(_translate("Dialog", "Шаг сканирования"))
        self.label_text_scan_step.setText(_translate("Dialog", "Угол [град.]"))
        self.lineEdit_scan_step.setText(_translate("Dialog", "15"))
        self.label_text_note_scan.setText(_translate("Dialog", "* Угол в 90 сканируется по умолчанию"))
        self.pushButton_processing.setText(_translate("Dialog", "Обработка"))
        self.label_text_deviation_original_and_received.setText(_translate("Dialog", "ℰ_org_received"))
import res_rc