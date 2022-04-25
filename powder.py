import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict

from main import plot_img, plot_dist, calc_moments, PIXELS_IN_CM
import numpy as np

#
# def find_circles(gray):
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.10, 10)
#     # ensure at least some circles were found
#     if circles is not None:
#         # convert the (x, y) coordinates and radius of the circles to integers
#         circles = np.round(circles[0, :]).astype("int")
#         output = img.copy()
#
#         # loop over the (x, y) coordinates and radius of the circles
#         for (x, y, r) in circles:
#             if 900 < y < 1100 and 750 < x < 1000:
#                 # draw the circle in the output image, then draw a rectangle
#                 # corresponding to the center of the circle
#                 cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#                 cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#         # show the output image
#         plt.imshow(output, cmap='gray')
#         plt.show()
#
#
# def circle_detector(img):
#     img = Image.open(direct + r"\\" + filename)
#     # Output image:
#     output_image = Image.new("RGB", img.size)
#     output_image.paste(img)
#     draw_result = ImageDraw.Draw(output_image)
#     # Find circles
#     rmin = 18
#     rmax = 20
#     steps = 100
#     threshold = 0.4
#     points = []
#     for r in range(rmin, rmax + 1):
#         for t in range(steps):
#             points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
#     acc = defaultdict(int)
#     for x, y in canny_edge_detector(img):
#         for r, dx, dy in points:
#             a = x - dx
#             b = y - dy
#             acc[(a, b, r)] += 1
#     circles = []
#     for k, v in sorted(acc.items(), key=lambda i: -i[1]):
#         x, y, r = k
#         if v / steps >= threshold and all(
#                 (x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
#             print(v / steps, x, y, r)
#             circles.append((x, y, r))
#     for x, y, r in circles:
#         draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))
#
#
# def sobel(filename):
#     from PIL import Image, ImageDraw
#     from math import sqrt
#
#     # Load image:
#     input_image = Image.open(filename)
#     input_pixels = input_image.load()
#
#     # Sobel kernels
#     kernely = [[-1, 0, 1],
#                [-2, 0, 2],
#                [-1, 0, 1]]
#     kernelx = [[-1, -2, -1],
#                [0, 0, 0],
#                [1, 2, 1]]
#
#     # Create output image
#     output_image = Image.new("RGB", input_image.size)
#     draw = ImageDraw.Draw(output_image)
#
#     # Compute convolution between intensity and kernels
#     for x in range(1, input_image.width - 1):
#         for y in range(1, input_image.height - 1):
#             magx, magy = 0, 0
#             for a in range(3):
#                 for b in range(3):
#                     xn = x + a - 1
#                     yn = y + b - 1
#                     intensity = sum(input_pixels[xn, yn]) / 3
#                     magx += intensity * kernelx[a][b]
#                     magy += intensity * kernely[a][b]
#
#             # Draw in black and white the magnitude
#             color = int(sqrt(magx ** 2 + magy ** 2))
#             draw.point((x, y), (color, color, color))
#
#     output_image.save("sobel.png")


def plot_distances(distance, center_x, center_y, img, original, cond, filename):
    plt.imshow(original, cmap='gray')
    if cond:
        plt.imshow(img, cmap='gray', alpha=0.1)
    x = np.flip(np.array([(400 + shift) for shift in range(0, 901, 150)]))
    # dist = np.array(range(424, 701, 55))
    y = np.sqrt(distance ** 2 - (x - center_x) ** 2) + center_y
    for i, dist in enumerate(distance):
        if i == len(x) - 1:
            x_loc = x[i] - 100
            x[i] += 10
        else:
            x_loc = x[i] - 70
        plt.plot([center_x, x[i]], [center_y, y[i]])
        plt.text(x_loc, y[i] + 90, np.round(dist), fontdict={'size': 8}, backgroundcolor='0.75')
    # plt.grid(True)
    plt.axis('off')
    plt.savefig(f'{filename}.png')
    plt.show()


def slice_img(image):
    return image[1000:2500, 350:1800, :]


if __name__ == '__main__':
    direct = r"D:\physics Exp\year 3\imprForExp1"
    directory = os.fsencode(direct)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            if filename.startswith("M_Debay"):
                img = slice_img(cv2.imread(direct + r"\\" + filename))
                orig_img = slice_img(cv2.imread(direct + r"\\" + filename[2:]))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray[gray < 230] = 0
                gray[gray >= 230] = 255
                x_, y_, r = [], [], []
                center_x, center_y = 800, 800
                # find contours in the binary image
                contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                no_center = True
                for c in contours:
                    cX, cY = calc_moments(c, img)
                    if 600 < cX < 900 and 600 < cY < 900 and no_center:
                        no_center = False
                        center_x = cX - 10
                        center_y = cY - 15
                    if not no_center:
                        r.append(np.sqrt(cv2.contourArea(c) / np.pi))
                    x_.append(cX)
                    y_.append(cY)
                # plot_dist(np.array(r)[1::2] + center_x, y_[1::2], center_x, center_y, gray)
                plot_distances(np.array(r)[1::2], center_x, center_y, gray, orig_img, True, "Debay with circles")
                plot_distances(np.array(r)[1::2], center_x, center_y, gray, orig_img, False, "Debay without circles")
