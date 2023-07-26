from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt
from astropy.io import fits
import sys
import os
import numpy as np
from ics.cobraCharmer import pfi as pfiControl

active_cobras = np.full(57, False)
bad_cobras = np.full(57, False)


class CobraButton(QPushButton):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.setText(str(idx+1))
        self.setCheckable(True)
        self.setMaximumWidth(20)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            active_cobras[self.idx] = not active_cobras[self.idx]
            super().mousePressEvent(QMouseEvent)
        elif QMouseEvent.button() == Qt.RightButton:
            bad_cobras[self.idx] = not bad_cobras[self.idx]
            if bad_cobras[self.idx]:
                self.setStyleSheet('background-color: #00FF0000')
            else:
                self.setStyleSheet('background-color: None')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pfi = pfiControl.PFI(fpgaHost='localhost', doLoadModel=False)

        self.setWindowTitle('Cobra controller')
        self.statusBar().showMessage('Please load XML file')

        block1 = QHBoxLayout()
        self.xml = QLineEdit()
        self.xml.setMinimumWidth(240)
        block1.addWidget(self.xml)
        self.btn_xml = QPushButton('Load XML file')
        self.btn_xml.clicked.connect(self.click_load)
        block1.addWidget(self.btn_xml)
        block1.addWidget(QSplitter(Qt.Vertical), QSizePolicy.Expanding)
        btn = QPushButton('Move Up/Down')
        btn.clicked.connect(self.move_up_down)
        block1.addWidget(btn)

        block2 = QHBoxLayout()
        self.btn_go = QPushButton('Go')
        block2.addWidget(self.btn_go, QSizePolicy.Expanding)
        self.btn_go.clicked.connect(self.click_go)
        onlyInt = QIntValidator()
        block2.addWidget(QLabel('theta:'))
        self.theta = QLineEdit('0')
        self.theta.setValidator(onlyInt)
        block2.addWidget(self.theta)
        block2.addWidget(QLabel('phi:'))
        self.phi = QLineEdit('0')
        self.phi.setValidator(onlyInt)
        block2.addWidget(self.phi)

        block3 = QHBoxLayout()
        btn_odd = QPushButton('Odd')
        btn_odd.setCheckable(True)
        btn_odd.clicked[bool].connect(self.click_odd)
        block3.addWidget(btn_odd)
        block4 = QHBoxLayout()
        btn_even = QPushButton('Even')
        btn_even.setCheckable(True)
        btn_even.clicked[bool].connect(self.click_even)
        block4.addWidget(btn_even)
        self.btn_cobras = []
        for idx in range(57):
            btn_c = CobraButton(idx)
            self.btn_cobras.append(btn_c)
            if idx % 2 == 0:
                block3.addWidget(btn_c, QSizePolicy.Minimum)
            else:
                block4.addWidget(btn_c, QSizePolicy.Minimum)
        btn_c = QPushButton()
        btn_c.setMaximumWidth(20)
        block4.addWidget(btn_c)

        block5 = QHBoxLayout()
        block5.addWidget(QLabel('Right click to mark bad cobras'), QSizePolicy.Expanding)
        self.cb_use_bad = QCheckBox('Use bad cobras')
        block5.addWidget(self.cb_use_bad)
        self.btn_speed = QPushButton('Fast')
        self.btn_speed.setCheckable(True)
        self.btn_speed.clicked[bool].connect(self.click_speed)
        block5.addWidget(self.btn_speed)

        layout = QVBoxLayout()
        layout.addLayout(block1)
        layout.addWidget(QSplitter())
        layout.addLayout(block2)
        layout.addLayout(block5)
        layout.addLayout(block3)
        layout.addLayout(block4)

        cw = QWidget()
        cw.setLayout(layout)
        self.setCentralWidget(cw)

    def click_speed(self, pressed):
        if pressed:
            self.btn_speed.setText('Slow')
        else:
            self.btn_speed.setText('Fast')

    def click_go(self):
        if not hasattr(self, 'all_cobras'):
            self.statusBar().showMessage('Load XML file first!')
            return
        cIdx = [idx for idx, c in enumerate(active_cobras) if c]
        if not self.cb_use_bad.isChecked():
            cIdx = [idx for idx in cIdx if not bad_cobras[idx]]
        if len(cIdx) <= 0:
            self.statusBar().showMessage('No cobras selected!')
            return

        use_fast = not self.btn_speed.isChecked()
        cobras = pfiControl.PFI.allocateCobraList(zip(np.full(len(cIdx), 1), np.array(cIdx) + 1))
        theta_steps = int(self.theta.text())
        phi_steps = int(self.phi.text())
        self.statusBar().showMessage('Start to moving cobra...')
        self.pfi.moveAllSteps(cobras, theta_steps, phi_steps, thetaFast=use_fast, phiFast=use_fast)
        self.statusBar().showMessage('Moving cobra succeed!')

    def click_odd(self, pressed):
        for idx in range(0, 57, 2):
            self.btn_cobras[idx].setChecked(pressed)
            active_cobras[idx] = pressed

    def click_even(self, pressed):
        for idx in range(1, 57, 2):
            self.btn_cobras[idx].setChecked(pressed)
            active_cobras[idx] = pressed

    def click_load(self):
        if len(self.xml.text()) <= 0:
            self.statusBar().showMessage(f"Error: enter XML filename!")
            return
        xml = '../xml/' + self.xml.text()
        if not os.path.exists(xml):
            self.statusBar().showMessage(f"Error: {xml} not presented!")
            return
        self.pfi.loadModel(xml)
        self.all_cobras = pfiControl.PFI.allocateCobraRange(range(1, 2))
        self.pfi.setFreq(self.all_cobras)
        self.btn_xml.setStyleSheet('background-color: green')

    def move_up_down(self):
        self.statusBar().showMessage(f"Function not implemented!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
