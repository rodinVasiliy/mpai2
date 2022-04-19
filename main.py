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
    df_dx = signal.convolve2d(img, vertical, mode='same', boundary='symm')
    df_dy = signal.convolve2d(img, horizontal, mode='same', boundary='symm')
    result = np.sqrt(df_dx ** 2 + df_dy ** 2)
    return df_dx, df_dy, result


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def laplacian_approximation(img, mask):

    laplacian = signal.convolve2d(img, mask, mode='same', boundary='symm')
    laplacian_estimation = np.abs(laplacian)
    return laplacian_estimation

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

    mask1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    mask2 = 1 / 2 * np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]])
    mask3 = 1 / 3 * np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    laplasian1 = laplacian_approximation(image, mask1)
    laplasian2 = laplacian_approximation(image, mask2)
    laplasian3 = laplacian_approximation(image, mask3)


    fig = plt.figure(figsize=(16, 10))

    fig.add_subplot(3, 4, 1)
    plt.title('Исходное изображение')
    imshow(image, cmap='gray')

    fig.add_subplot(3, 4, 2)
    plt.title('Оценка модуля лапласиана 1')
    imshow(laplasian1, cmap='gray')

    fig.add_subplot(3, 4, 3)
    plt.title('Гистограмма оценки модуля лапласиана 1')
    hist1, bins1 = histogram(laplasian1)
    plt.plot(bins1, hist1)

    fig.add_subplot(3, 4, 4)
    plt.title('Контуры метода аппроксимации лапласиана 1')
    laplacian_approximation_image1 = thresholding(laplasian1, 85)
    imshow(laplacian_approximation_image1, cmap='gray')

    fig.add_subplot(3, 4, 6)
    plt.title('Оценка модуля лапласиана 2')
    imshow(laplasian2, cmap='gray')

    fig.add_subplot(3, 4, 7)
    hist2, bins2 = histogram(laplasian2)
    plt.title('Гистограмма оценки модуля лапласиана 2 ')
    plt.plot(bins2, hist2)

    fig.add_subplot(3, 4, 8)
    plt.title('Контуры метода аппроксимации лапласиана 2')
    laplacian_approximation_image2 = thresholding(laplasian2, 45)
    imshow(laplacian_approximation_image2, cmap='gray')

    fig.add_subplot(3, 4, 10)
    plt.title('Оценка модуля лапласиана 3')
    imshow(laplasian3, cmap='gray')

    fig.add_subplot(3, 4, 11)
    hist3, bins3 = histogram(laplasian3)
    plt.title('Гистограмма оценки модуля лапласиана 3')
    plt.plot(bins3, hist3)

    fig.add_subplot(3, 4, 12)
    plt.title('Контуры метода аппроксимации лапласиана 3')
    laplacian_approximation_image3 = thresholding(laplasian3, 65)
    imshow(laplacian_approximation_image3, cmap='gray')

    plt.show()




