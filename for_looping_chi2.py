import numpy as np
from ReliefF import ReliefF
import pandas as pd
from SBS import SBS

# x = x.transpose()
#importing a classifier for testing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


#importing the dataset
complete_testing_data = pd.read_csv('C:\\Users\\Dewesh\\PycharmProjects\\Minor2.1\\dataset\\testing.csv')
complete_training_data = pd.read_csv('C:\\Users\\Dewesh\\PycharmProjects\\Minor2.1\\dataset\\training.csv')

#print(complete_training_data.iloc[0:4,0:6])

#modification of the dataset into array like structure
ti = complete_training_data.iloc[:,1:].values
tl = complete_training_data.iloc[:,0].values
vi = complete_testing_data.iloc[:,1:].values   #insert testing
vl = complete_testing_data.iloc[:,0].values    #insert testing
nearest_neighbours = 9
num = 7

ti1 = SelectKBest(mutual_info_classif, k=10).fit(ti,tl).transform(ti)
vi1 = SelectKBest(mutual_info_classif, k=10).fit(ti,tl).transform(vi)



fs = ReliefF(n_neighbors=nearest_neighbours,n_features_to_keep=num)
X_train = fs.fit_transform(ti,tl)

#print(fs.top_features[0:20])
#print(complete_training_data.iloc[0:4,fs.top_features[0:10]])


threshold = 0.98
unsuitable_attributes = set()
suitable = list()
i=0
j=0
for i in range (0,50):
    if(fs.top_features[i] in unsuitable_attributes ):
        continue
    suitable.append(fs.top_features[i])
    for j in range(i+1,50):
        #print(np.corrcoef(ti[:, 60], ti[:, 18]))
        cov = np.corrcoef(ti[:,fs.top_features[i]],ti[:,fs.top_features[j]])[0][1]
        if(cov>threshold):
            unsuitable_attributes.add(fs.top_features[j])

print(fs.top_features[0:7])
print(suitable[0:7])

#print(ti[:,[60,81]])

feature_selector = SBS(estimator=GaussianNB(),k_features=10)
some_fs_variable=feature_selector.fit(ti[:,suitable[0:30]],tl)

new_data = some_fs_variable.transform(ti[:,suitable[0:30]])

valid_data = some_fs_variable.transform(vi[:,suitable[0:30]]) #vi[:,suitable[0:30]]
#print(new_data[0,:])






#knn classifier
"""
knn1 = KNeighborsClassifier(n_neighbors=7)
knn1.fit(ti[:,[18,60,144,81,18,102,123]], tl)
pred1 = knn1.predict(vi[:,[18,60,144,81,18,102,123]])
acc1 = accuracy_score(pred1, vl)
index = 0
'''
for x in pred1:
    print('Actual :',vl[index],'Predicted :',pred1[index])
    index = index+1
'''
print(acc1)
"""


#SVM classifier
"""
SVM=SVC()
SVM.fit(ti[:,suitable[0:num]], tl)
pred_SVM = SVM.predict(vi[:,suitable[0:num]])
acc_SVM = accuracy_score(vl, pred_SVM)
index = 0
'''
for x in pred_SVM:
    print('Actual :',vl[index],'Predicted :',pred_SVM[index])
    index = index+1
'''
print(acc_SVM)
"""

num =20
ft =  [81,60,18,39]
#NB classifier only relief
#"""
print('\nOnly relief applied..  10 attributes')
gnb = GaussianNB()
pred_nb = gnb.fit(ti[:,fs.top_features[0:10]], tl).predict(vi[:,fs.top_features[0:10]])
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)
#"""


#Decision tree only relief
#"""
clf = tree.DecisionTreeClassifier()
clf.fit(ti[:,fs.top_features[0:10]],tl)
pred_dt  = clf.predict(vi[:,fs.top_features[0:10]])
acc_dt = accuracy_score(vl,pred_dt)
print(acc_dt)
#"""

#NB classifier  relief + redundancy removal
#"""
print('\nRelief + redundancy removal..  10 attributes')
gnb = GaussianNB()
pred_nb = gnb.fit(ti[:,suitable[0:10]], tl).predict(vi[:,suitable[0:10]])
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)
#"""


#Decision tree only relief
#"""
clf = tree.DecisionTreeClassifier()
clf.fit(ti[:,suitable[0:10]],tl)
pred_dt  = clf.predict(vi[:,suitable[0:10]])
acc_dt = accuracy_score(vl,pred_dt)
print(acc_dt)
#"""



#testiung all applied
#print(some_fs_variable.indices_)
print("\nAll fs meathods applied..  10 attributes")
#"""
pred_nb = gnb.fit(new_data, tl).predict(valid_data)
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)

clf.fit(new_data,tl)
pred_dt  = clf.predict(valid_data)
acc_dt = accuracy_score(vl,pred_dt)
print(acc_dt)

#"""

#testiung nothing applieed all
#print(some_fs_variable.indices_)
print("\nAll 147 attributes.. nothing applied")
#"""
pred_nb = gnb.fit(ti, tl).predict(vi)
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)

clf.fit(ti,tl)
pred_dt  = clf.predict(vi)
acc_dt = accuracy_score(vl,pred_dt)
print(acc_dt)

#"""



#pca
pca = PCA(n_components=10)
pca.fit(ti)
X_test = pca.fit_transform(vi)
X_train = pca.fit_transform(ti)
print('\nPCA applied.. 10 attributes selected')
gnb = GaussianNB()
pred_nb = gnb.fit(X_train,tl).predict(X_test)
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)

data_reduction = 13700/147
accuracy_reduction_nb = 76.5285996055 - 75.7396449704

print('\nNB classifier results..')
print('after application of all')
print('Dataset reduction :',data_reduction,'%','    Acuuracy chnage : -',accuracy_reduction_nb,'%')

'''
clf.fit(new_data,tl)
pred_dt  = clf.predict(valid_data[:,some_fs_variable.indices_])
acc_dt = accuracy_score(vl,pred_dt)
print(acc_dt)
'''


print('\nmutual_info 15')
gnb = GaussianNB()
pred_nb = gnb.fit(ti1, tl).predict(vi1)
acc_nb = accuracy_score(vl, pred_nb)
print(acc_nb)