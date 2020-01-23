import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn import  neighbors
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


#suppress all FutureWarnings
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#dataset--1
#df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#df.replace('?',-99999, inplace=True)
#df.drop(['id'], 1, inplace=True)
#X = np.array(df.drop(['class'], 1))
#y = np.array(df['class'])

#dataset --2
#data = pd.read_csv('data.csv');
#remove last column
#data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
#features_mean= list(data.columns[1:31])
#X1 = data.loc[:,features_mean] #input
#y = data.loc[:, 'diagnosis'] #target
#scaler=MinMaxScaler()
#X=scaler.fit_transform(X1) #input

#dataset---3
data = load_breast_cancer()
label_names = data['target_names'] 
y = data['target'] #target
feature_names = data['feature_names'] 
X = data['data'] # data


train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state = 42)
#matrix of all results
accuracy_all = []
cvs_all = []

# 1-Initialize GaussianNB classifier
start = time.time()
clf = GaussianNB()
# Train our classifier
model = clf.fit(train, train_labels)
# Make predictions
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("1-GaussianNB accuracy:",clf.score(test,test_labels))
print("1-GaussianNB accuracy :",accuracy_score(prediction,test_labels))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# 2-Initialize our dummy classifier
start = time.time()
dummy=DummyClassifier()
# Train our classifier
dummy.fit(train, train_labels)
prediction = dummy.predict(test)
scores = cross_val_score(dummy, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("2-Dummy accuracy:",dummy.score(test,test_labels))
print("2-Dummy Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# 3-Initialize our KNeighbors classifier
start = time.time()
clf = neighbors.KNeighborsClassifier()
# Train our classifier
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("3-KNeighbors accuracy:",clf.score(test,test_labels))
print("3-KNeighbors Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# 4-Initialize our KNeighbors classifier
start = time.time()
rf = RandomForestClassifier()
# Train our classifier
rf.fit(train, train_labels)
prediction = rf.predict(test)
scores = cross_val_score(rf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("4-RandomForest accuracy:",rf.score(test,test_labels))
print("4-RandomForest Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# 5-Initialize LogisticRegression classifier
start = time.time()
lr = LogisticRegression()
# Train our classifier
lr.fit(train, train_labels)
prediction = lr.predict(test)
scores = cross_val_score(lr, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("5-logistic regression accuracy:",lr.score(test,test_labels))
print("5-logistic regression Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# 6-Initialize support vector machine classifier
start = time.time()
svc_linear = SVC()
# Train our classifier
svc_linear.fit(train, train_labels)
prediction = svc_linear.predict(test)
scores = cross_val_score(svc_linear, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("6-support vector machine accuracy:",svc_linear.score(test,test_labels))
print("6-support vector machine Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))



# 7-Initialize  NuSVC support vector machine classifier
start = time.time()
clf = NuSVC()
# Train our classifier
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(svc_linear, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("7- NuSVC support vector machine accuracy:",clf.score(test,test_labels))
print("7-NuSVC support vector machine Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))



# 8-Initialize  LinearSVC support vector machine classifier
start = time.time()
clf = LinearSVC()
# Train our classifier
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(svc_linear, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print("8-LinearSVC support vector machine accuracy:",clf.score(test,test_labels))
print("8-LinearSVC support vector machine Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

#9-Initialize SGD classifier

start = time.time()
clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction,test_labels))
cvs_all.append(np.mean(scores))

#print(" 9-SGD accuracy:",clf.score(test,test_labels))
print("9-SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

#10-Initialize MLP classifier

start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(8,),alpha=1, max_iter=1000)
#mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, random_state=1,learning_rate_init=.1)
mlp.fit(train, train_labels)
prediction = mlp.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, test_labels))
cvs_all.append(np.mean(scores))

#print(" 10-MLP accuracy:",mlp.score(test,test_labels))
print("10-MLP Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


#11-Initialize AdaBoost classifier

start = time.time()
clf = AdaBoostClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, test_labels))
cvs_all.append(np.mean(scores))

#print(" 11-AdaBoost accuracy:",clf.score(test,test_labels))
print("11-AdaBoost Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

#12-Initialize GaussianProcess classifier

start = time.time()
clf=GaussianProcessClassifier(1.0 * RBF(1.0))
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, test_labels))
cvs_all.append(np.mean(scores))

#print(" 12-GaussianProcess accuracy:",clf.score(test,test_labels))
print("12-GaussianProcess Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


#13-Initialize DecisionTree  Classifier

start = time.time()
clf=DecisionTreeClassifier(max_depth=5)
clf.fit(train, train_labels)
prediction = clf.predict(test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, test_labels))
cvs_all.append(np.mean(scores))

#print(" 13-DecisionTree accuracy:",clf.score(test,test_labels))
print("13-DecisionTree Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction,test_labels)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))



x=list(np.array(accuracy_all))




















