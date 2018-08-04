import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random
import sys
import os

from PyQt5.QtWidgets import QApplication, QDialog, QGridLayout, QProgressBar, QLabel
from skimage import novice
from skimage import color
from skimage import io
from skimage.filters import sobel


class Tracing(QDialog):
    def __init__(self, main_img, acc_img, initial, ban, name, parent):
        super(Tracing, self).__init__(parent)
        self.main_img = main_img
        self.acc_img = acc_img
        self.name = name

        self.initial = initial
        self.ban = ban

        self.N = float(0)
        self.edge_size = 0
        self.runs = float(0)
        self.FFT = None
        self.dist_dict = dict()
        self.path = []

        self.apricot = (float(252) / float(255), float(200) / float(255), float(155) / float(255))
        self.magenta = (float(255) / float(255), float(95) / float(255), float(162) / float(255))
        self.gray = (float(220) / float(255), float(220) / float(255), float(220) / float(255))
        self.green = (float(0) / float(255), float(255) / float(255), float(65) / float(255))

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 12))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.ax.set_facecolor('k')
        self.margin_x = 0
        self.margin_y = 0
        self.max_x = 0
        self.max_y = 0

        self.trace = None
        self.drawer_x = None
        self.drawer_y = None
        self.pin = None
        self.h_tracer = None
        self.v_tracer = None

        self.collection = None
        self.patches_x = None
        self.patches_y = None

        self.layout = QGridLayout(self)
        self.setGeometry(900, 100, 800, 120)
        self.setWindowTitle('Epicycle Drawing')

        self.path_progress = QProgressBar(self)
        self.trace_progress = QProgressBar(self)

        self.path_label = QLabel(self)
        self.trace_label = QLabel(self)

        self.path_label.setText('Path construction progress:')
        self.trace_label.setText('Tracing Progress:')

        self.layout.addWidget(self.path_label, 0, 0)
        self.layout.addWidget(self.trace_label, 1, 0)
        self.layout.addWidget(self.path_progress, 0, 1)
        self.layout.addWidget(self.trace_progress, 1, 1)

        self.show()

    def init_outline(self):
        # Take input image
        inp = io.imread(self.main_img)
        im = color.rgb2gray(inp)

        # Get coordinates of points in edge
        edges = sobel(im)
        indices = np.where(edges > 0.375)

        x_s = indices[1]
        y_s = indices[0]

        self.edge_size = len(x_s)
        self.path_progress.setMaximum(len(x_s))

        # Flip image
        picture = novice.open(self.main_img)
        self.max_x = int(picture.size[0])
        self.max_y = int(picture.size[1])
        self.margin_x = int(picture.size[0])
        self.margin_y = int(picture.size[1])

        for i in range(self.edge_size):
            y_s[i] = self.max_y - y_s[i]

        # Combine coordinate lists
        xy_list = zip(x_s, y_s)

        # Create closed loop path
        self.path = self.create_path(list(xy_list), self.initial, self.ban)

        # Compute FFT of points in edge
        self.N = float(len(self.path))
        self.runs = float(len(self.path))
        self.trace_progress.setMaximum(self.runs)

        x_list, y_list = zip(*self.path)

        edge_points = np.zeros((1, len(self.path)), dtype=np.complex_)

        for i in range(len(self.path)):
            edge_points[0][i] = x_list[i] + y_list[i]*1j

        self.FFT = np.fft.fft(edge_points, n=int(self.N))

    def draw_acc_img(self, max_y):
        # Draw accessory image
        inp_acc = io.imread(self.acc_img)
        im_acc = color.rgb2gray(inp_acc)

        edges_acc = sobel(im_acc)
        indices_acc = np.where(edges_acc > 0.375)

        x_acc = indices_acc[1]
        y_acc = indices_acc[0]

        for i in range(len(y_acc)):
            y_acc[i] = max_y - y_acc[i]

        self.ax.plot([x_acc], [y_acc], marker='o', mfc=self.magenta, mec=self.magenta, ms=1, ls='None')

    def init_trace(self):
        self.trace, = self.ax.plot([], [], marker='o', mfc=self.magenta, mec=self.magenta, ms=1, ls='None')
        self.h_tracer, = self.ax.plot([], [], c=self.green, lw=1)
        self.v_tracer, = self.ax.plot([], [], c=self.green, lw=1)
        self.drawer_x, = self.ax.plot([], [], c=self.green, marker='o', mfc=self.green, mec=self.green, ms=5, lw=1.5)
        self.drawer_y, = self.ax.plot([], [], c=self.green, marker='o', mfc=self.green, mec=self.green, ms=5, lw=1.5)
        self.pin, = self.ax.plot([], [], 'wo', ms=5)

        x_sum = 0
        y_sum = 0

        x_partial = []
        y_partial = []

        circle_atts = []
        self.patches_x = []
        self.patches_y = []

        for k in range(int(self.N)):
            x_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.cos(np.angle(self.FFT[0][k]))
            x_partial.append(x_sum)
            y_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.sin(np.angle(self.FFT[0][k]))
            y_partial.append(y_sum)

        for k in range(1, int(self.N)):
            circle_atts.append(float(1 / self.N) * np.abs(self.FFT[0][k]))

        for i in range(len(circle_atts)):
            h = x_partial[i]
            k = y_partial[i]
            r = circle_atts[i]
            circle_x = plt.Circle((h, k + self.margin_y), r, ec=self.apricot, fc='none', ls='dashed', lw=2)
            circle_y = plt.Circle((h + self.margin_x, k), r, ec=self.apricot, fc='none', ls='dashed', lw=2)
            self.patches_x.append(circle_x)
            self.patches_y.append(circle_y)

        for j in range(len(self.patches_x)):
            self.ax.add_artist(self.patches_x[j])
            self.ax.add_artist(self.patches_y[j])

        return self.trace, self.drawer_x, self.drawer_y, self.pin, self.h_tracer, self.v_tracer

    def trace_dots(self, i):   # Plt FFT of points in edge
        x_sum = 0
        y_sum = 0

        x_partial_dx = []
        y_partial_dx = []
        x_partial_dy = []
        y_partial_dy = []

        for k in range(int(self.N)):
            speed = 2 * np.pi * k / self.N
            x_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.cos(speed * i + np.angle(self.FFT[0][k]))
            x_partial_dx.append(x_sum)
            x_partial_dy.append(x_sum + self.margin_x)
            y_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.sin(speed * i + np.angle(self.FFT[0][k]))
            y_partial_dx.append(y_sum + self.margin_y)
            y_partial_dy.append(y_sum)

        self.trace, = self.ax.plot([x_sum], [y_sum], marker='o', mfc=self.magenta, mec=self.magenta, ms=1, ls='None')
        self.pin.set_data([x_sum], [y_sum])
        self.h_tracer.set_data([x_sum, x_sum + self.margin_x], [y_sum, y_sum])
        self.v_tracer.set_data([x_sum, x_sum], [y_sum, y_sum + self.margin_y])
        self.drawer_x.set_data(x_partial_dx, y_partial_dx)
        self.drawer_y.set_data(x_partial_dy, y_partial_dy)

        for s in range(len(self.patches_x)):
            h = x_partial_dx[s]
            k = y_partial_dy[s]
            self.patches_x[s].center = h, k + self.margin_y

        for s in range(len(self.patches_y)):
            h = x_partial_dx[s]
            k = y_partial_dy[s]
            self.patches_y[s].center = h + self.margin_x, k

        self.trace_progress.setValue(i + 1)
        QApplication.processEvents()

        return self.trace, self.drawer_x, self.drawer_y, self.pin, self.h_tracer, self.v_tracer

    def create_path(self, point_list, initial, ban):
        path = []
        if initial == 'random':
            initial = random.choice(point_list)
        else:
            path.append(initial)

        while len(path) != (len(point_list) - len(ban)):
            distances = dict()

            for i in range(len(point_list)):
                if (point_list[i] in path) is False:
                    distances[point_list[i]] = self.distance(point_list[i], initial)
            nxt = min(distances, key=distances.get)

            if nxt not in ban:
                path.append(nxt)

            self.path_progress.setValue(len(path) + len(ban))
            QApplication.processEvents()

            initial = nxt

        return path

    def distance(self, point1, point2):
        if (point1, point2) in self.dist_dict or (point2, point1) in self.dist_dict:
            if (point1, point2) in self.dist_dict:
                return self.dist_dict[(point1, point2)]
            elif (point2, point1) in self.dist_dict:
                return self.dist_dict[(point2, point1)]
        else:
            dx = float(point1[0]) - float(point2[0])
            dy = float(point1[1]) - float(point2[1])
            dist = np.sqrt(dx ** 2 + dy ** 2)
            self.dist_dict[(point1, point2)] = dist
            return dist

    def save_video(self):
        self.init_outline()

        self.ax.set_xlim(-150, 2*self.max_x + 150)
        self.ax.set_ylim(-150, 2*self.max_y + 150)

        self.ax.set_visible('False')

        if self.acc_img is not None:
            self.draw_acc_img(self.max_y)

        frames = int(self.runs)
        video = anim.FuncAnimation(self.fig, self.trace_dots, init_func=self.init_trace, frames=frames, interval=10, blit=True)

        if not os.path.exists('test/videos'):
            os.mkdir('test/videos')

        video.save('test/videos/' + self.name + '.mp4', writer='ffmpeg')

        print "Video saved."

        self.close()

    def save_image(self):
        self.init_outline()

        x_processed = []
        y_processed = []

        for i in range(int(self.runs)):
            x_sum = 0
            y_sum = 0

            for k in range(int(self.N)):
                speed = 2 * np.pi * k / self.N
                x_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.cos(speed * i + np.angle(self.FFT[0][k]))
                y_sum += float(1 / self.N) * np.abs(self.FFT[0][k]) * np.sin(speed * i + np.angle(self.FFT[0][k]))

            x_processed.append(x_sum)
            y_processed.append(y_sum)

            self.trace_progress.setValue(i + 1)
            QApplication.processEvents()

        self.ax.set_xlim(0, 2 * self.max_x)
        self.ax.set_ylim(0, 2 * self.max_y)

        self.ax.scatter(x_processed, y_processed, s=0.5, c=self.magenta, marker="o")

        plt.show()

        if not os.path.exists('test/images'):
            os.mkdir('test/images')

        self.fig.savefig('test/images/' + self.name + '.png', dpi=1080)

        print "Image saved."

        self.close()