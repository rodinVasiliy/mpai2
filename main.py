import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
from skimage.exposure import histogram
from scipy import signal


def thresholding(image, threshold):
    thresholding_result = image > threshold
    thresholding_result = (thresholding_result * 255)
    return thresholding_result


def simple_gradient(img):
    vertical = np.array([[-1], [1]])
    horizontal = np.array([[-1, 1]])
    u = signal.convolve2d(img, vertical, mode='same', boundary='symm')
    v = signal.convolve2d(img, horizontal, mode='same', boundary='symm')
    result = np.sqrt(u ** 2 + v ** 2)
    return u, v, result


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'images//07_elaine.tif'
    image = imread(path)
    u, v, gradient = simple_gradient(image)
    simple_gradient_image = thresholding(gradient, 35)

    fig = plt.figure(figsize=(16, 10))

    fig.add_subplot(2, 3, 1)
    plt.title('Исходное изображение')
    imshow(image, cmap='gray')

    fig.add_subplot(2, 3, 2)
    plt.title('Частная производная по первому направлению')
    imshow(u+128, cmap='gray')

    fig.add_subplot(2, 3, 3)
    plt.title('Частная производная по второму направлению')
    imshow(v+128, cmap='gray')

    fig.add_subplot(2, 3, 4)
    plt.title('Оценка модуля градиента')
    imshow(gradient, cmap='gray')

    fig.add_subplot(2, 3, 5)
    hist, bins = histogram(gradient)
    plt.title('Гистограмма оценки модуля градиента')
    plt.plot(bins, hist)

    fig.add_subplot(2, 3, 6)
    plt.title('Контуры метода простого градиента')
    imshow(simple_gradient_image, cmap='gray')

    plt.show()





