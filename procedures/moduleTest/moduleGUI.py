from PyQt5.QtWidgets import (
    QPushButton, QMainWindow, QHBoxLayout, QLineEdit, QSplitter,
    QSizePolicy, QLabel, QCheckBox, QVBoxLayout, QWidget, QApplication)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from moduleTest import ModuleTest, getCobras

active_cobras = np.full(57, False)
bad_cobras = np.full(57, False)
IP = '128.149.77.24'
#IP = '133.40.164.251'
camSplit = 25

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
    mt = None
    pfi = None

    def __init__(self):
        super().__init__()
        self.control_btn = []
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
        self.cb_refresh = QCheckBox('auto refresh')
        block1.addWidget(self.cb_refresh)
        btn = QPushButton('Check positions')
        self.control_btn.append(btn)
        btn.clicked.connect(self.check_positions)
        block1.addWidget(btn)

        block2 = QHBoxLayout()
        btn = QPushButton('Forward')
        self.control_btn.append(btn)
        block2.addWidget(btn, QSizePolicy.Expanding)
        btn.clicked.connect(lambda: self.click_go(1))
        onlyInt = QIntValidator()
        block2.addWidget(QLabel('theta:'))
        self.theta = QLineEdit('0')
        self.theta.setValidator(onlyInt)
        block2.addWidget(self.theta)
        block2.addWidget(QLabel('phi:'))
        self.phi = QLineEdit('0')
        self.phi.setValidator(onlyInt)
        block2.addWidget(self.phi)
        btn = QPushButton('Reverse')
        self.control_btn.append(btn)
        block2.addWidget(btn, QSizePolicy.Expanding)
        btn.clicked.connect(lambda: self.click_go(-1))

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
        for idx in range(56, -1, -1):
            if idx % 2 == 0:
                block3.addWidget(self.btn_cobras[idx], QSizePolicy.Minimum)
            else:
                block4.addWidget(self.btn_cobras[idx], QSizePolicy.Minimum)
        btn_c = QPushButton()
        btn_c.setMaximumWidth(20)
        block4.addWidget(btn_c)

        block5 = QHBoxLayout()
        block5.addWidget(QLabel('Right click to mark bad cobras'), QSizePolicy.Expanding)
        self.cb_use_bad = QCheckBox('bad cobras')
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

    def click_go(self, direction=1):
        if self.mt is None:
            self.statusBar().showMessage('Load XML file first!')
            return
        cIdx = [idx for idx, c in enumerate(active_cobras) if c]
        if not self.cb_use_bad.isChecked():
            cIdx = [idx for idx in cIdx if not bad_cobras[idx]]
        if len(cIdx) <= 0:
            self.statusBar().showMessage('No cobras selected!')
            return

        for btn in self.control_btn:
            btn.setEnabled(False)
        use_fast = not self.btn_speed.isChecked()
        cobras = getCobras(cIdx)
        theta_steps = int(self.theta.text()) * direction
        phi_steps = int(self.phi.text()) * direction
        self.statusBar().showMessage('Start to moving cobra...')
        self.pfi.moveAllSteps(cobras, theta_steps, phi_steps, thetaFast=use_fast, phiFast=use_fast)
        self.statusBar().showMessage('Moving cobra succeed!')
        if self.cb_refresh.isChecked():
            self.check_positions()
        for btn in self.control_btn:
            btn.setEnabled(True)

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
        xml = self.xml.text()
        if not os.path.exists(xml):
            self.statusBar().showMessage(f"Error: {xml} is not presented!")
            return
        if self.mt is not None:
            self.statusBar().showMessage(f"Error: {xml} is already set!")
            return

        self.mt = ModuleTest(IP, xml, brokens=None, camSplit=camSplit)
        self.pfi = self.mt.pfi
        self.btn_xml.setStyleSheet('background-color: green')
        self.btn_xml.setEnabled(False)

    def check_positions(self):
        """ show current cobra arm angles """
        if self.mt is None:
            self.statusBar().showMessage('Load XML file first!')
            return
        brokens = (np.arange(57) + 1)[bad_cobras]
        self.mt.setBrokenCobras(brokens)

        data1 = self.mt.cam1.expose()
        data2 = self.mt.cam2.expose()
        pos = self.mt.extractPositions(data1, data2)

        tht, phi, _ = self.pfi.positionsToAngles(self.mt.goodCobras, pos)
        c = self.pfi.calibModel.centers[self.mt.goodIdx]
        t = self.pfi.thetaToGlobal(self.mt.goodCobras, tht[:,0])
        p = t + phi[:,0] + self.pfi.calibModel.phiIn[self.mt.goodIdx]
        L1 = self.pfi.calibModel.L1[self.mt.goodIdx]
        L2 = self.pfi.calibModel.L2[self.mt.goodIdx]
        s = self.mt.goodIdx[self.mt.goodIdx<=camSplit].shape[0]
        T1 = self.pfi.calibModel.tht0[self.mt.goodIdx]
        T2 = self.pfi.calibModel.tht1[self.mt.goodIdx]

        plt.close()
        plt.figure(figsize=(15,6))
        plt.subplot(211)
        ax = plt.gca()
        ax.axis('equal')

        ax.plot(c[:s].real, c[:s].imag, 'r.')
        ax.plot(pos[:s].real, pos[:s].imag, 'bo')

        p1 = c[:s]
        p2 = p1 + L1[:s]*np.exp(t[:s]*(1j))
        p3 = p2 + L2[:s]*np.exp(p[:s]*(1j))
        t1 = p1 + L1[:s]*np.exp(T1[:s]*(1j))
        t2 = p1 + L1[:s]*np.exp(T2[:s]*(1j))
        ax.plot(t1.real, t1.imag, 'y.')
        ax.plot(t2.real, t2.imag, 'y.')
        for n in range(s):
            ax.plot([p1[n].real, p2[n].real], [p1[n].imag, p2[n].imag], 'g', linewidth=1)
            ax.plot([p2[n].real, p3[n].real], [p2[n].imag, p3[n].imag], 'c', linewidth=1)
            ax.text(c[n].real-20, c[n].imag-30, f'#{self.mt.goodIdx[n]+1}')
            ax.text(c[n].real-20, c[n].imag-50, f'{np.rad2deg(t[n]):.1f}')
            ax.text(c[n].real-20, c[n].imag-70, f'{np.rad2deg((p-t)[n]+np.pi):.1f}')

        plt.subplot(212)
        ax = plt.gca()
        ax.axis('equal')

        ax.plot(c[s:].real, c[s:].imag, 'r.')
        ax.plot(pos[s:].real, pos[s:].imag, 'bo')

        p1 = c[s:]
        p2 = p1 + L1[s:]*np.exp(t[s:]*(1j))
        p3 = p2 + L2[s:]*np.exp(p[s:]*(1j))
        t1 = p1 + L1[s:]*np.exp(T1[s:]*(1j))
        t2 = p1 + L1[s:]*np.exp(T2[s:]*(1j))
        ax.plot(t1.real, t1.imag, 'y.')
        ax.plot(t2.real, t2.imag, 'y.')
        for n in range(len(self.mt.goodIdx)-s):
            ax.plot([p1[n].real, p2[n].real], [p1[n].imag, p2[n].imag], 'g', linewidth=1)
            ax.plot([p2[n].real, p3[n].real], [p2[n].imag, p3[n].imag], 'c', linewidth=1)
            ax.text(c[s+n].real-20, c[s+n].imag-30, f'#{self.mt.goodIdx[s+n]+1}')
            ax.text(c[s+n].real-20, c[s+n].imag-50, f'{np.rad2deg(t[s+n]):.1f}')
            ax.text(c[s+n].real-20, c[s+n].imag-70, f'{np.rad2deg((p-t)[s+n]+np.pi):.1f}')

        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
