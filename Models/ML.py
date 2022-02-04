from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def MLModel(modelSTR):
  dictML = {"KNN": KNeighborsClassifier(),
            "R-Forest": RandomForestClassifier(n_jobs=1, max_depth=10),
            "D-Tree": DecisionTreeClassifier()}
  model = dictML.get(modelSTR)
  return model
