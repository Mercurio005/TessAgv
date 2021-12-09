import os
import cv2 as cv
import numpy as np
from TessAgv.Utils.GeneratorClass import VIgenerator

def LoadData(path, bands, size):
  tifAll, maskAll = loadFiles(path, bands)
  X, y = allPatches(tifAll, maskAll, size)
  return X, y

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

def allPatches(tifList, maskList, size):
  X, y = createPatches(tifList[0], maskList[0], size)
  for i in range(1, len(tifList)):
    Xi, yi, = createPatches(tifList[i], maskList[i], size)
    X = X + Xi
    y = y + yi
  return X, y

def createPatches(tif, mask, size):
  patchesTIF = list()
  patchesMask = list()
  newY = (mask.shape[0] // size[0])*size[0]
  newX = (mask.shape[1] // size[1])*size[1]
  diffY = mask.shape[0] - newY
  diffX = mask.shape[1] - newX
  initY = diffY//2
  initX = diffX//2
  newMask = mask[initY:initY+newY, initX:initX+newX]
  newTif = tif[initY:initY+newY, initX:initX+newX]

  for y in range(0, newTif.shape[0], size[0]):
    for x in range(0, newTif.shape[1], size[1]):
      windowTIF = newTif[y:y + size[0], x:x + size[1]]
      windowMask = newMask[y:y + size[0], x:x + size[1]]
      if windowTIF.shape[0] != size[0] or windowTIF.shape[1] != size[1]:
        continue
      patchesTIF.append(windowTIF)
      patchesMask.append(np.expand_dims(windowMask, 2))
  return patchesTIF, patchesMask

def getBands(tif, bands):
  imageList = list()
  for band in bands:
    dataBand = tif.getBand(band)
    if np.isnan(dataBand).any():
      dataBand = np.nan_to_num(dataBand, None)
    imageList.append(dataBand)
  imageAll = cv.merge(imageList)
  return imageAll
