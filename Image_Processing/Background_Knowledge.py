import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure

# # Load the image
path = "src.JPG"
ref = cv.imread(path)
path2 = "ref.JPG"
src = cv.imread(path2)
matched = exposure.match_histograms(src, ref, multichannel='multi')

# Resize the image
scale_percent = 20  # percent of original size
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)
dim = (width, height)
src = cv.resize(src, dim, interpolation=cv.INTER_AREA)
ref = cv.resize(ref, dim, interpolation=cv.INTER_AREA)
resized = cv.resize(matched, dim, interpolation=cv.INTER_AREA)

# Show the resized image
cv.imshow("Src", src)
cv.waitKey(0)
cv.imshow("Ref", ref)
cv.waitKey(0)
cv.imshow('Resized image', resized)
cv.imwrite("Match.jpg", resized)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imwrite('gray.jpg', gray)
# Obtain the histogram
hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(src.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

equ = cv.equalizeHist(gray)

cv.imshow('Equalized image', equ)
cv.imwrite('Equalized_image.jpg', equ)
cv.waitKey(0)
cv.destroyAllWindows()

# construct a figure to display the histogram plots for each channel
# before and after histogram matching was applied
(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
# loop over our source image, reference image, and output matched
# image
for (i, image) in enumerate((src, ref, matched)):
    # convert the image from BGR to RGB channel ordering
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # loop over the names of the channels in RGB order
    for (j, color) in enumerate(("red", "green", "blue")):
        # compute a histogram for the current channel and plot it
        (hist, bins) = exposure.histogram(image[..., j],
                                          source_range="dtype")
        axs[j, i].plot(bins, hist / hist.max())
        # compute the cumulative distribution function for the
        # current channel and plot it
        (cdf, bins) = exposure.cumulative_distribution(image[..., j])
        axs[j, i].plot(bins, cdf)
        # set the y-axis label of the current plot to be the name
        # of the current color channel
        axs[j, 0].set_ylabel(color)
# set the axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")
# display the output plots
plt.tight_layout()
plt.show()
