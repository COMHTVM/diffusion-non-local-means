import cv2
import scipy as sc
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]

image = cv2.imread("/home/sarvesh/Projects/Image Processing/lenna.noise.jpg",
                   0)
image = image.astype('float')

def anisotropic(image, n, delta, kappa):
    # Initialise the output container with the given image
    out = image
    out = out.astype('float64')
    dd = np.sqrt(2)

    # Initialise gradient vectors along 8 directions
    hN = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64)
    hS = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64)
    hE = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64)
    hW = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64)
    hNE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64)
    hSE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64)
    hSW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64)
    hNW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64)

    for i in range(n):
        # Convolution with gradient vectors
        nN = ndimage.filters.convolve(out, hN)
        nS = ndimage.filters.convolve(out, hS)
        nE = ndimage.filters.convolve(out, hE)
        nW = ndimage.filters.convolve(out, hW)
        nNE = ndimage.filters.convolve(out, hNE)
        nSE = ndimage.filters.convolve(out, hSE)
        nSW = ndimage.filters.convolve(out, hSW)
        nNW = ndimage.filters.convolve(out, hNW)

        # Diffusion rates
        cN = 1. / (1 + (nN / kappa)**2)
        cS = 1. / (1 + (nS / kappa)**2)
        cE = 1. / (1 + (nE / kappa)**2)
        cW = 1. / (1 + (nW / kappa)**2)
        cNE = 1. / (1 + (nNE / kappa)**2)
        cSE = 1. / (1 + (nSE / kappa)**2)
        cSW = 1. / (1 + (nSW / kappa)**2)
        cNW = 1. / (1 + (nNW / kappa)**2)

        out = out + delta * (
            (cN * nN) + (cE * nE) + (cS * nS) +
            (cW * nW)) + delta * (1 / (dd**2)) * (
                (cNE * nNE) + (cSE * nSE) + (cSW * nSW) + (cNW * nNW))
    return out

def isotropic(lenna, steps, lambd):
    # Defining a laplacian window of 3*3
    window = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Make a image border of width 1
    img = cv2.copyMakeBorder(lenna, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Initialise the final image
    final = img

    # Iterations given by variable steps
    for k in range(steps):

        # Run through each pixel of image
        for i in range(lenna[0].size):
            for j in range(lenna[0].size):
                final[i:i + 3, j:j +
                      3] = final[i:i + 3, j:j +
                                 3] + lambd * final[i + 1, j + 1] * window
    return final

def gaussian(l, sig):
    # Generate array
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    # Generate 2D matrices by duplicating ax along two axes
    xx, yy = np.meshgrid(ax, ax)
    # kernel will be the gaussian over the 2D arrays
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    # Normalise the kernel
    final = kernel / kernel.sum()
    return final

def means_filter(image):
    [m, n] = image.shape

    # Padding the image
    img = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

    # Empty output image
    out = np.zeros((m, n), dtype='float')

    # generate gaussian kernel matrix of 7*7
    kernel = gaussian(7, 1)
    h = 25
    h = h * h

    # Run the non-local means for each pixel
    for i in range(6, 512):
        for j in range(6,512):
            w1 = img[i:i + 7, j:j + 7]

            wmax = 0
            avg = 0
            sweight = 0
            rmin = i - 2
            rmax = i + 2
            cmin = j - 2
            cmax = j + 2

            # Apply Gaussian weighted square distance between 
            # patches of 7*7 in a window of 5*5
            for r in range(rmin, rmax):
                for c in range(cmin, cmax):
                    w2 = img[r - 3:r + 4, c - 3:c + 4]
                    bl = w1 - w2
                    temp = np.multiply(bl, bl)
                    d = sum(sum(np.multiply(kernel, temp)))

                    w = np.exp(d / h)
                    if w > wmax:
                        wmax = w
                    sweight = sweight + w
                    avg = avg + w * img[r, c]

            avg = avg + wmax * image[i, j]
            sweight = sweight + wmax
            if sweight > 0:
                out[i-5 , j-5 ] = avg / sweight
            else:
                out[i-5 , j-5 ] = image[i, j]

    return out

# Call means_filter for the input image
means = means_filter(image)

# Call isotropic function with parameters:
# Iterations =10, Lambda = 0.2
iso = isotropic(image, 10, 0.2)

# Call the Anisotropic function with the parameters:
# Iterations = 10, Delta = 0.14, Kappa = 15
aniso = anisotropic(image, 10, 0.14, 15)

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Original')
plt.subplot(2, 2, 2), plt.imshow(aniso, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('After Anisotropic fitlering')
plt.subplot(2, 2, 3), plt.imshow(iso, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('After Isotropic filtering')
plt.subplot(2, 2, 4), plt.imshow(means, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('After Non-local means denoising')

plt.show()