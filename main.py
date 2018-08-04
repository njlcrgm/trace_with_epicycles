from epicycles import *

import time
import sys
import os

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton


class ControlPanel(QWidget):
    def __init__(self):
        super(ControlPanel, self).__init__()

        self.layout = QGridLayout(self)
        self.setGeometry(100, 100, 800, 120)
        self.setWindowTitle('Epicycle Drawing')

        self.button_main_anim = QPushButton('Save Main Animation', self)
        self.button_acc_anim = QPushButton('Save Accessory Animation', self)
        self.button_main_out = QPushButton('Save Main Output', self)
        self.button_acc_out = QPushButton('Save Accessory Output', self)

        self.button_main_anim.clicked.connect(self.main_anim)
        self.button_acc_anim.clicked.connect(self.acc_anim)
        self.button_main_out.clicked.connect(self.main_out)
        self.button_acc_out.clicked.connect(self.acc_out)

        self.layout.addWidget(self.button_main_anim, 0, 0)
        self.layout.addWidget(self.button_acc_anim, 0, 1)
        self.layout.addWidget(self.button_main_out, 1, 0)
        self.layout.addWidget(self.button_acc_out, 1, 1)

    def main_anim(self):
        ma = Tracing('inputs/twice.jpg', 'inputs/twice_teardrop.jpg', (530, 338), [(272, 209)], 'main', self)
        ma.save_video()

    def acc_anim(self):
        aa = Tracing('inputs/twice_teardrop.jpg', None, (361, 171), [], 'acc', self)
        aa.save_video()

    def main_out(self):
        mo = Tracing('inputs/twice.jpg', 'inputs/twice_teardrop.jpg', (530, 338), [(272, 209)], 'main', self)
        mo.save_image()

    def acc_out(self):
        ao = Tracing('inputs/twice_teardrop.jpg', None, (361, 171), [], 'acc', self)
        ao.save_image()

def main():
    app = QApplication(sys.argv)
    panel = ControlPanel()

    if not os.path.exists('test'):
        os.mkdir('test')
        time.sleep(3)

    panel.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
