import os
import cv2 as cv
import numpy as np
from GeneratorClass import VIgenerator

def loadFiles(cPath, bands):
  iterator = 1
  maskList = list()
  tifList = list()
  pathTIF = cPath + "/TIFF/Agaves{}.tif".format(iterator)
  pathMask = cPath + "/Mask/Mask{}.png".format(iterator)
  while os.path.exists(pathTIF):
    mask = cv.imread(pathMask, cv.IMREAD_GRAYSCALE)
    maskList.append(mask)
    tifRaw = VIgenerator(pathTIF)
    tif = getBands(tifRaw, bands)
    tifList.append(tif)
    iterator += 1
    pathTIF = cPath + "/TIFF/Agaves{}.tif".format(iterator)
    pathMask = cPath + "/Mask/Mask{}.png".format(iterator)
  return tifList, maskList
