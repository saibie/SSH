import os
import sys
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
#         self.setLayout(self.layout)
        self.setGeometry(200, 200, 1500, 600)
        
    def initUI(self):
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.appfig = plt.Figure()
        self.appcanvas = FigureCanvas(self.appfig)
        
        self.treeview = QTreeView()
        self.listview = QListView()
        
        path = '../../Optic_Images'
        if not os.path.isdir(path):
            path = '.'
        
        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.dirModel.setFilter(QDir.NoDotDot | QDir.AllDirs)
        
        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot |  QDir.Files)

        self.treeview.setModel(self.dirModel)
        self.listview.setModel(self.fileModel)

        self.treeview.setRootIndex(self.dirModel.index(path))
        self.listview.setRootIndex(self.fileModel.index(path))

        self.treeview.clicked.connect(self.on_clicked)
        
        self.loadButton = QPushButton('Load')
        self.loadButton.clicked.connect(self.load_optic)
        self.applyButton = QPushButton('Apply')
        self.applyButton.clicked.connect(self.doGraph2)
        
        hbox00 = QHBoxLayout()
        vbox00 = QVBoxLayout()
        vbox00.addWidget(self.treeview)
        vbox00.addWidget(self.listview)
        vbox00.addWidget(self.loadButton)
        hbox00.addLayout(vbox00)

        hbox00.addWidget(self.canvas)
#         cb = QComboBox()
#         cb.addItem('Graph1')
#         cb.addItem('Graph2')
#         cb.activated[str].connect(self.onComboBoxChanged)
        vbox02 = QVBoxLayout()
#         vbox02.addWidget(cb)
        
        self.dsb_blurpower = QDoubleSpinBox()
        self.dsb_blurpower.setRange(1, 20)
        self.dsb_blurpower.setSingleStep(.01)
        self.dsb_blurpower.setDecimals(2)
        
        self.dsb_centerhall = QDoubleSpinBox()
        self.dsb_centerhall.setRange(0, 5)
        self.dsb_centerhall.setSingleStep(.1)
        self.dsb_centerhall.setDecimals(1)
        
        self.dsb_axis1 = QDoubleSpinBox()
        self.dsb_axis1.setRange(0, 1000)
        self.dsb_axis1.setSingleStep(1)
        self.dsb_axis1.setDecimals(0)
        
        self.dsb_axis2 = QDoubleSpinBox()
        self.dsb_axis2.setRange(0, 1000)
        self.dsb_axis2.setSingleStep(1)
        self.dsb_axis2.setDecimals(0)
        
        self.dsb_axissame = QCheckBox('장단축 동일')
        
        self.dsb_angle = QDoubleSpinBox()
        self.dsb_angle.setRange(0, 360)
        self.dsb_angle.setSingleStep(.1)
        self.dsb_angle.setDecimals(1)
        
        self.dsb_xcenter = QDoubleSpinBox()
        self.dsb_xcenter.setRange(0, 1000)
        self.dsb_xcenter.setSingleStep(1)
        self.dsb_xcenter.setDecimals(0)
        
        self.dsb_ycenter = QDoubleSpinBox()
        self.dsb_ycenter.setRange(0, 1000)
        self.dsb_ycenter.setSingleStep(1)
        self.dsb_ycenter.setDecimals(0)
        
        gbox = QGroupBox('변수 조절')
        self.grid = QGridLayout()
        self.grid.addWidget(QLabel('Blur 강도 :'), 0, 0)
        self.grid.addWidget(self.dsb_blurpower, 0, 1)
        self.grid.addWidget(QLabel('중앙홀 :'), 1, 0)
        self.grid.addWidget(self.dsb_centerhall, 1, 1)
        self.grid.addWidget(QLabel('1축 :'), 2, 0)
        self.grid.addWidget(self.dsb_axis1, 2, 1)
        self.grid.addWidget(QLabel('2축 :'), 3, 0)
        self.grid.addWidget(self.dsb_axis2, 3, 1)
        self.grid.addWidget(self.dsb_axissame, 4, 0, 1, 0)
        self.grid.addWidget(QLabel('회전각 :'), 5, 0)
        self.grid.addWidget(self.dsb_angle, 5, 1)
        self.grid.addWidget(QLabel('가로중심 :'), 6, 0)
        self.grid.addWidget(self.dsb_xcenter, 6, 1)
        self.grid.addWidget(QLabel('세로중심 :'), 7, 0)
        self.grid.addWidget(self.dsb_ycenter, 7, 1)
        
        gbox.setLayout(self.grid)
    
        vbox02.addWidget(gbox)
        vbox02.addWidget(self.applyButton)
        hbox00.addLayout(vbox02)
        
        hbox00.addWidget(self.appcanvas)
        
        self.setLayout(hbox00)
        
#         self.onComboBoxChanged(cb.currentText())
        
#     def onComboBoxChanged(self, text):
#         if text == 'Graph1':
#             self.doGraph1()
#         elif text == 'Graph2':
#             self.doGraph2()
        
    def load_optic(self):
        self.fig.clear()
        filepath = self.fileModel.fileInfo(self.listview.currentIndex()).absoluteFilePath()
        try:
            imA = Image.open(filepath)
            self.A = np.asarray(imA) / 255
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.imshow(self.A)
            self.ax.axis('off')
            self.canvas.draw()
        except:
            warn = QMessageBox.information(self, 'Error', 'Image Loading Error.')
        
    def doGraph1(self):
        x = np.arange(0, 10, 0.5)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(x, y1, label="sin(x)")
        ax.plot(x, y2, label="cos(x)", linestyle="--")
        
        ax.set_xlabel("x")
        ax.set_xlabel("y")
        
        ax.set_title("sin & cos")
        ax.legend()
        
        self.canvas.draw()
    def doGraph2(self):
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        Z = X**2 + Y**2
        
        self.fig.clear()
        
        ax = self.fig.gca(projection='3d')
        ax.plot_wireframe(X, Y, Z, color='black')
        self.canvas.draw()
        
        
    def on_clicked(self, index):
        path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listview.setRootIndex(self.fileModel.setRootPath(path))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()