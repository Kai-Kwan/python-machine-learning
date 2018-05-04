
# Import the necessary modules and libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pandas import read_csv
import math


# Program init function
def init(ratio , filename):
    # Read data file
    data = read_csv(filename)
    # Rename the data for easy to use
    data.rename(columns={'Pregnancies': 0, 'Glucose': 1, 'BloodPressure': 2, 'SkinThickness': 3, 'Insulin': 4, 'BMI': 5,
                         'DiabetesPedigreeFunction': 6, 'Age': 7, 'Outcome': 8}, inplace=True)
    # Define the training ratio
    trainingRate = ratio
    trainSize = int(len(data)*trainingRate)
    # Extrect the test set
    X_testSet = (data[trainSize:int(len(data))])
    # Pure the data by changing it to nan value and drop the row of it
    #data.drop(data.head(0).index, inplace=True)
    #data = data.dropna(subset=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    #data = data.reset_index(drop=True)

    # The mdata is the modified data
    mData = read_csv(filename)
    mData.rename(columns={'Pregnancies': 0, 'Glucose': 1, 'BloodPressure': 2, 'SkinThickness': 3, 'Insulin': 4, 'BMI': 5,
                         'DiabetesPedigreeFunction': 6, 'Age': 7, 'Outcome': 8}, inplace=True)
    mData.drop(mData.head(0).index, inplace=True)
    mData = mData.dropna(subset=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    mData = mData.reset_index(drop=True)
    mData[[1,2,3,4,5,6,7]] = mData[[1,2,3,4,5,6,7]].replace(0,np.nan).astype(float)
    for i in range(1,7):
        mData[i] = mData[i].replace(mData > mData[i].mean(skipna=True)*1.9 , np.nan)
        mData[i] = mData[i].replace(mData < mData[i].mean(skipna=True)*0.1, np.nan)

    mData = mData.dropna(subset = [0,1,2,3,4,5,6,7,8])
    mData = mData.reset_index(drop=True)

    # Create training data
    trainData=mData
    trainDataSet = (trainData[0:trainSize])
    featureSelection = [0,1,2,3,4,5,6,7]

    # Fiting the data to the random forest X is the training data y in the targeted outcome
    X = trainDataSet[featureSelection]
    y = trainDataSet[8]
    regr_1 = RandomForestClassifier(n_estimators=400,max_depth=12,max_features = 5,min_samples_leaf=1, min_samples_split=2, random_state=10)
    regr_1.fit(X, y);

    # predict the outcome using the test set
    y_1 = regr_1.predict(X_testSet[featureSelection])

    # Calculate the accuracy by compering the outputs in predict and test set
    ans = X_testSet[8]
    num_correct = 0
    for i in range(0,len(ans)):
        ans1 = int(y_1[i])
        ans2 = int(ans[i+trainSize])
        if ans1 == ans2:
            num_correct = num_correct + 1

    print('%s of %s test values correct.'%(num_correct, len(X_testSet)))
    accuracy=num_correct/len(X_testSet)
    print('accuracy',accuracy)
    print("\n")

    #cross_val in testset
    X = X_testSet[featureSelection]
    y = X_testSet[8]
    print("In cross val in test data")
    acc= cross_val_score(regr_1,X,y,scoring='accuracy');
    print(acc)
    print(acc.mean())
    print("\n")

    #cross_val in training set
    X = trainDataSet[featureSelection]
    y = trainDataSet[8]
    print("In cross val in training data")
    acc= cross_val_score(regr_1,X,y,scoring='accuracy');
    print(acc)
    print(acc.mean())
    print("\n")

    #plot the feature importances
    features = trainDataSet[[0,1,2,3,4,5,6,7]]
    importances = regr_1.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()

#testing differen ratio of training data and test data
def printDifferentRatio(filename):
    for ratio in range(1,10):
        print(ratio)
        init(ratio*0.1, filename)
        print("\n")

# main() program
# can change the training ratio to build the model
trainingRatio = 0.5
#file name for the data, the dta file mus plase in the same directory as this file
filename = 'diabetes.csv'
init(trainingRatio , filename)

#printDifferentRatio(filename) # un common this to see the prints of the training and testing ratio from 1-10
