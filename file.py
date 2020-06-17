import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import seaborn as sns
plt.show()



names = ['a','b','c','d','e','f','g','h','i','j','k']
data1 = pd.read_csv(r"C:\Users\Or\Downloads\NormalizedData.csv" ,header=None,names=names,nrows=400)
data1.head()
data = data1.sort_index(ascending=True,axis=0)
data.head()
feat = ['a','b','c','d','e','f','g','h','i','j']
X = data[feat]
Y = data.k
#logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
cnf_metrix = metrics.confusion_matrix(y_test,y_pred)
print("truePositive:",cnf_metrix[0][0])
print("falseNegative:",cnf_metrix[0][1])
print("falsePositive:",cnf_metrix[1][0])
print("trueNegative:",cnf_metrix[1][1])
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#random forest
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

