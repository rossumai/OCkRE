from ockre import OCkRE
from synthset import CropImageIterator
import fakestrings
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
import numpy as np


ocr = OCkRE()
ocr.loadweights('densified_labeltype_best.h5')
source = CropImageIterator()

for i in range(5):
    sample = source.next()
    ans = ocr.ocr_frompic(image = sample[0], labeltype = sample[3])
    im = Image.fromarray((sample[0].reshape(512,64).T * 255).astype(np.uint8))
    plt.title("Gold label: ['%s'], Decoded: %s" % (sample[1],ans))
    plt.imshow(im,cmap='gray',norm=NoNorm(),interpolation="none")
    plt.savefig("%s.png" % sample[1])