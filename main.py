import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.io.wavfile
from scipy import signal
from sklearn.cluster import MiniBatchKMeans

PIXELS_IN_CM = 480.77


def find_center(name):
    """

    :param name:
    :return:
    """
    pass


#
# def conv_der(im):
#     """
#     computes the magnitude of image derivatives. You should derive the image in each direction separately (vertical and
#     horizontal) using simple convolution with [0.5, 0, -0.5] as a row and column vectors. Next, use these derivative
#     images to compute the magnitude image
#     :param im: grayscale images of type float64
#     :return: the magnitude of the derivative, with the same dtype and shape.
#     """
#     kernel = np.array([0.5, 0, -0.5]).reshape((3, 1))
#     dy = scipy.signal.convolve2d(im, kernel, mode="same")
#     dx = scipy.signal.convolve2d(im, kernel.T, mode="same")
#     return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
#

# Press the green button in the gutter to run the script.
def quantize_gray(image, n_clusters=10):
    (h, w) = image.shape[:2]
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=n_clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # image = image.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    return quant


#
# def bin():
#     for file in os.listdir(directory):
#         filename = os.fsdecode(file)
#         if filename.endswith(".jpg"):
#             # reading image
#             img = cv2.imread(direct + r"\\" + filename)
#             if filename.startswith("KI"):
#                 img = img[900:2500, 700:2200, :]
#                 continue
#             if filename.startswith("M_Li_F"):
#                 img = img[500:3000, :2500, :]
#             if filename.startswith("NaCl"):
#                 img = img[1000:2400, 500:2100, :]
#                 continue
#             # converting image into grayscale image
#
#             # setting threshold of gray image
#             # _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
#             # using a findContours() function
#             # contours, _ = cv2.findContours(
#             #     threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             # edges = cv2.Canny(gray, threshold1=60, threshold2=80)
#             # quantize = quantize_gray(img, 5)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
#             morphology_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
#             plt.imshow(morphology_img, 'Greys_r')
#
#             plt.imshow(gray, cmap='gray')
#             plt.show()
#             if filename.startswith("KI"):
#                 gray[gray == 103] = gray[gray.shape]
#                 gray[gray == 120] = gray[gray.shape]
#             if filename.startswith("M_Li_F"):
#                 pass
#             if filename.startswith("NaCl"):
#                 gray[gray == 105] = gray[gray.shape]
#             # plt.imshow(gray, cmap='gray')
#             # plt.show()
#             edges = conv_der(gray)
#             # edges[edges <= 4] = 0
#             plt.imshow(edges, cmap='gray')
#             plt.show()
#             #
#             # i = 0
#             #
#             # # list for storing names of shapes
#             # for contour in contours:
#             # # here we are ignoring first counter because
#             # # find_contour function detects whole image as shape
#             #     if i == 0:
#             #         i = 1
#             #         continue
#             #
#             #     # cv2.approxPloyDP() function to approximate the shape
#             #     approx = cv2.approxPolyDP(
#             #         contour, 0.01 * cv2.arcLength(contour, True), True)
#             #
#             #     # using drawContours() function
#             #     cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
#             #
#             #     # finding center point of shape
#             #     M = cv2.moments(contour)
#             #     if M['m00'] != 0.0:
#             #         x_p = int(M['m10'] / M['m00'])
#             #         y_p = int(M['m01'] / M['m00'])
#             #     else:
#             #         x_p, y_p = 0, 0
#             #
#             #     # putting shape name at center of each shape
#             #     if len(approx) == 3:
#             #         cv2.putText(img, 'Triangle', (x_p, y_p),
#             #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             #
#             #     elif len(approx) == 4:
#             #         cv2.putText(img, 'Quadrilateral', (x_p, y_p),
#             #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             #
#             #     elif len(approx) == 5:
#             #         cv2.putText(img, 'Pentagon', (x_p, y_p),
#             #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             #
#             #     elif len(approx) == 6:
#             #         cv2.putText(img, 'Hexagon', (x_p, y_p),
#             #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             #
#             #     else:
#             #         cv2.putText(img, 'circle', (x_p, y_p),
#             #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             #
#             # # displaying the image after drawing contours
#             # cv2.imshow('shapes', img)
#             #
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#
#
# # def find_circles(gray, output):
# #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10)
# #     if circles is not None:
# #         # # convert the (x_p, y_p) coordinates and radius of the circles to integers
# #         # circles = np.round(circles[0, :]).astype("int")
# #         # loop over the (x_p, y_p) coordinates and radius of the circles
# #         for (x, y, r) in circles:
# #             # draw the circle in the output image, then draw a rectangle
# #             # corresponding to the center of the circle
# #             cv2.circle(output, (x, y), r, (0, 255, 0), 4)
# #             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# #         # show the output image
# #         cv2.imshow("output", np.hstack([img, output]))
# #         cv2.waitKey(0)
#

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    result[result < 20] = result[500, 500]
    return result


def calc_z(x_p, y_p):
    z_dist = 1.5 * PIXELS_IN_CM
    tan_2t = np.sqrt(x_p ** 2 + y_p ** 2) / z_dist
    theta = np.arctan(tan_2t) / 2
    return np.sqrt(x_p ** 2 + y_p ** 2) * np.tan(theta)


def plot_dist(x, y, center_x, center_y, image, name):
    plt.imshow(image, cmap='gray')
    dist = (np.sqrt((np.array(x) - center_x) ** 2 + (np.array(y) - center_y) ** 2))
    plt.title(f"distances from center, {name}")
    for i in range(len(x)):
        shift_x = -130
        shift_y = -40
        if x[i] > img.shape[1] / 2:
            shift_x = -50
        if y[i] > img.shape[0] / 2:
            shift_y = 80

        plt.plot([center_x, x[i]], [center_y, y[i]])
        plt.text(x[i] + shift_x, y[i] + shift_y, np.round(dist[i]), fontdict={'size': 8})
    plt.axis('off')
    plt.savefig(f'distances from center, {name}.png')
    plt.show()


def plot_coordinates(x, y, image, name, center1, center2):
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, linewidths=0.01, marker='*', color='r')
    plt.title(f"coordinates, {name}")
    for i, j in zip(x, y):
        shift_x = -150
        shift_y = -40
        if i > img.shape[1] / 2:
            shift_x = -50
        if j > img.shape[0] / 2:
            shift_y = 60

        plt.text(i + shift_x, j + shift_y, f"({i - center1}, {- (j - center2)})", fontdict={'size': 8})
    plt.axis('off')
    plt.savefig(f'coordinates, {name}.png')
    plt.show()


def plot_miller(x, y, center_x, center_y, img, name):
    plt.imshow(img, cmap='gray')
    # plt.title(f"angle: {angle}")
    plt.title(f"miller indices, {name}")
    for i in range(len(x)):
        plt.plot([center_x, x[i]], [center_y, y[i]])
        x_q = abs(x[i] - center_x)
        y_q = abs(y[i] - center_y)
        z_q = calc_z(x[i], y[i])
        minimal = min(x_q, y_q, z_q)
        shift_x = -100
        shift_y = -40
        if x[i] > img.shape[1] / 2:
            shift_x = -50
        if y[i] > img.shape[0] / 2:
            shift_y = 60
        if minimal != 0:
            x_q, y_q, z_q = np.floor(x_q / minimal).astype(int), \
                            np.floor(y_q / minimal).astype(int), \
                            np.floor(z_q / minimal).astype(int)
            plt.text(x[i] + shift_x, y[i] + shift_y, f"h:k:L={x_q}:{y_q}:{z_q}", fontdict={'size': 5.5})
    plt.axis('off')
    plt.savefig(f'miller indices, {name}.png')
    plt.show()


def plot_img(thresh, name, grid=True):
    plt.imshow(thresh, cmap='gray')
    plt.locator_params(axis='x_p', nbins=10)
    plt.locator_params(axis='y_p', nbins=10)
    if grid:
        plt.grid(True)
    plt.axis('off')
    plt.savefig(f'{name}.png')
    plt.show()


# def clean_image(img):
#     # blur
#     blur = cv2.GaussianBlur(img, (3, 3), 0)
#
#     # convert to hsv and get saturation channel
#     sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:, :, 1]
#
#     # threshold saturation channel
#     thresh = cv2.threshold(sat, 50, 255, cv2.THRESH_BINARY)[1]
#
#     # apply morphology close and open to make mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
#     # mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
#
#     # do OTSU threshold to get circuit image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
#     return otsu
#

def calc_moments(shape, img):
    # calculate moments for each contour
    M = cv2.moments(shape)
    # calculate x_p,y_p coordinate of center
    x = int(M["m10"] / M["m00"])  # if M["m00"] != 0 else 0
    y = int(M["m01"] / M["m00"])  # if M["m00"] != 0 else 0
    # print(f"centroid: X:{x}, Y:{y}")
    cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return x, y


if __name__ == '__main__':
    # iterate through all the images in the folder
    direct = r"D:\physics Exp\year 3\imprForExp1\original images"
    directory = os.fsencode(direct)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            # reading image
            # if filename.startswith("Li_F"):
            if filename.startswith("M_LiF"):
                # for angle in np.arange(7, 10.1, 0.5):
                angle = 7
                img = cv2.imread(direct + r"\\" + filename)[500:2800, :2300, :]
                orig = cv2.imread(direct + r"\\" + filename[2:])[500:2800, :2300, :]
                # img = img[330:3455, :2580, :]
                # output = img.copy()
                min_x, max_x = 1000, 1200
                min_y, max_y = 1100, 1400
            elif filename.startswith("M_NaCl"):
                angle = -0.5
                img = cv2.imread(direct + r"\\" + filename)[1000:2500, 500:2200, :]
                orig = cv2.imread(direct + r"\\" + filename[2:])[1000:2500, 500:2200, :]
                min_x, max_x = 700, 900
                min_y, max_y = 500, 700
            elif filename.startswith("M_KI"):
                angle = 9
                img = cv2.imread(direct + r"\\" + filename)[1000:2400, 700:2200, :]
                orig = cv2.imread(direct + r"\\" + filename[2:])[1000:2400, 700:2200, :]
                min_x, max_x = 700, 900
                min_y, max_y = 500, 700
            else:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            gray = rotate_image(gray, angle)
            orig = rotate_image(orig, angle)
            # gray = gray[gray > 127]
            gray = np.max(gray) - gray
            # gray = gray > 70
            # gray = gray.astype(int)
            ret, thresh = cv2.threshold(gray, 60, 255, 0)
            # find_circles(gray, output)

            X_s, Y_s = [], []
            center_x, center_y = 0, 0
            # find contours in the binary image
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 200000:
                    continue
                cX, cY = calc_moments(c, img)
                if min_x < cX < max_x and min_y < cY < max_y:
                    center_x = cX
                    center_y = cY
                X_s.append(cX)
                Y_s.append(cY)

            salt = filename.split("_")[1]
            plot_dist(X_s, Y_s, center_x, center_y, orig, salt)
            # plot_coordinates(X_s, Y_s, orig, salt, center_x, center_y)
            # plot_miller(X_s, Y_s, center_x, center_y, orig, salt)
