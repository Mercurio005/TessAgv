from TessAgv.Utils.Loader import LoadData
from TessAgv.Models import ML, DeepLearning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow import keras
import numpy as np

class ModeDL:
  def __init__(self, path, bandsNames, imageSize, epochs):
    self.__path = path
    self.__bands = bandsNames
    self.__size = imageSize
    self.__epochs = epochs
    self.__model = None
    self.__callbacks = callbacks = [
               keras.callbacks.ModelCheckpoint("agaves_segmentation.h5", save_best_only=True)]
    
  def train(self):
    X, y = LoadData(self.__path, self.__bands, self.__size)
    X_train, X_valid, y_train, y_valid = train_test_split(np.array(X), np.array(y), test_size = 0.3)
    self.__model = DeepLearning.U_Net(X_train[0].shape, len(np.unique(y_train)), X_train)
    self.__model.fit(X_train, y_train, epochs=self.__epochs, validation_data=(X_valid, y_valid), callbacks=self.__callbacks)
    
    
class ModeML:
  def __init__(self, path, bandsNames, imageSize, method):
    self.__path = path
    self.__bands = bandsNames
    self.__size = imageSize
    self.__model = ML.MLModel(method)
    self.__pca = None
    
  def train(self):
    X, y = LoadData(self.__path, self.__bands, self.__size)
    X_train, X_valid, y_train, y_valid = train_test_split(np.array(X), np.array(y), test_size = 0.3)
    Xnew, ynew = self.__transformData(X_train, y_train)
    self.__model.fit(Xnew, ynew)
    
  def __transformData(self, X_train, y_train):
    newX_train = X_train.reshape(X_train.shape[0], -1)
    newy_train = y_train.reshape(y_train.shape[0], -1)
    self.__pca = PCA()
    self.__pca.fit_transform(newX_train)

    # Calculating optimal k to have 95% (say) variance 
    k = 0
    total = sum(self.__pca.explained_variance_)
    current_sum = 0
    while(current_sum / total < 0.99):
        current_sum += self.__pca.explained_variance_[k]
        k += 1
    #k=20
    self.__pca = PCA(n_components=k, whiten=True)
    x_train_pca = self.__pca.fit_transform(newX_train)
    return x_train_pca, newy_train
