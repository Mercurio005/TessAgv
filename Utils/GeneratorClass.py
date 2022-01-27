import rasterio
import cv2 as cv
import numpy as np

class VIgenerator:
  def __init__(self, path):
    self.__TIF = rasterio.open(path)
    self.__Bands = self.__TIF.descriptions
    self.__wavelength = [650, 560, 450]
    self.__channelsList = ["Blue", "Green", "Red", "NIR", "RedEdge"]
    self.__switcher = {"NDVI": self.__NDVI(), 
                       "ARVI": self.__ARVI(), 
                       "GNDVI": self.__GNDVI(), 
                       "NDRE": self.__NDRE(), 
                       "MSAVI": self.__MSAVI(),
                       "NDWI": self.__NDWI()}

  @property
  def mask(self):
    tempMask = self.__TIF.read(self.__Bands.index(None)+1)
    tempMask[tempMask==0] = np.nan
    tempMask[tempMask==255] = 1
    return tempMask

  def __singleBand(self, band):
    return self.__TIF.read(self.__Bands.index(band)+1)

  def __NDVI(self):
    return cv.divide(cv.subtract(self.__singleBand("NIR"), self.__singleBand("Red")), cv.add(self.__singleBand("NIR"), self.__singleBand("Red")))

  def __ARVI(self):
    return cv.divide((self.__singleBand("NIR")-(2*self.__singleBand("Red") - self.__singleBand("Blue"))), (self.__singleBand("NIR")+(2*self.__singleBand("Red") - self.__singleBand("Blue"))))

  def __GNDVI(self):
    return cv.divide(cv.subtract(self.__singleBand("NIR"), self.__singleBand("Green")), cv.add(self.__singleBand("NIR"), self.__singleBand("Green")))

  def __NDRE(self):
    return cv.divide(cv.subtract(self.__singleBand("NIR"), self.__singleBand("RedEdge")), cv.add(self.__singleBand("NIR"), self.__singleBand("RedEdge")))

  def __MSAVI(self):
    return ((2*self.__singleBand("NIR") + 1 - np.sqrt(np.power(2*self.__singleBand("NIR") + 1,2) - 8*(self.__singleBand("NIR")-self.__singleBand("Red"))))*(1/2)) * self.mask

  def __NDWI(self):
    return cv.divide(cv.subtract(self.__singleBand("Green"), self.__singleBand("NIR")), cv.add(self.__singleBand("Green"), self.__singleBand("NIR")))

  def getBand(self, bandName):
    if bandName in self.__channelsList:
      return self.__singleBand(bandName) * self.mask
    elif bandName in self.__switcher:
      return self.__switcher.get(bandName)
    elif bandName == "RGB":
      return cv.merge([self.__singleBand("Blue")*self.__wavelength[2] * self.mask,
                       self.__singleBand("Green")*self.__wavelength[1] * self.mask, 
                       self.__singleBand("Red")*self.__wavelength[0]] * self.mask)
