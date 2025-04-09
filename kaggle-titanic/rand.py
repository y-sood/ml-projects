import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def normaliser(A):
    for field in A:
        if field in ['Age']:
            A[field] = (A[field]/A[field].abs().max())
        else: 
            continue
    return A

def clean(A):
    #Dropping irrelevant fields
    A= A.drop(['Name','Embarked','Ticket','Cabin', 'Fare', 'SibSp', 'Parch'],axis=1)
    #Removing rows with NA values
    A = A.dropna()
    #Dealing with categorical variables
    A['Sex'] = A['Sex'].replace({'male': 1, 'female': 0})
    return A

def crinptr(A):
    X = A[['Sex', 'Age', 'Pclass']].to_numpy()
    Y = A[['Survived']].to_numpy()
    return X,Y

def crinpts(A):
    X_test = A[['Sex', 'Age', 'Pclass']].to_numpy()
    Y_test = []
    Z = A['PassengerId'].to_numpy()
    return X_test, Y_test, Z

def plotdes(A,B,C,D):
    #Scatter
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(A, B, C, c = D, marker = '^', cmap = "nipy_spectral")
    plt.title('Survivability in the titanic')
    ax.set_xlabel('Sex');ax.set_ylabel('Age');ax.set_zlabel('Pclass')
    #Show
    plt.show()

#TRAINING
#Reading in training data
hndl = open("titanic/train.csv")
dftr = pd.read_csv(hndl)
#Normalising
dftr_normal = normaliser(dftr)
dftr_normal = clean(dftr_normal)
#Creating input and output fields
X, Y = crinptr(dftr_normal)
#Splitting into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Classifier
clf = RandomForestClassifier()
clf.fit(X_train,Y_train.ravel())

#TESTING
Y_pred = (clf.predict(X_test))
#PLOT
#plotdes((X_test[:,0]), (X_test[:,1]), (X_test[:,2]), Y_test)

#Output data
#Reading in answer data
hndl = open("titanic/test.csv")
dfts = pd.read_csv(hndl)
#Normalising
dfts_normal = normaliser(dfts)
dfts_normal = clean(dfts_normal)
#Creating answer input fields
X_test, Y_test, passid = crinpts(dfts_normal)
#Storing output
Y_test = (clf.predict(X_test))
#Output format
df_out = pd.DataFrame()
df_out['PassengerId'] = passid.tolist()
df_out['Survived'] = Y_test.tolist()
df_out.to_csv("gendersubmission.csv")




