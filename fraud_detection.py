# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:20:33 2019
@author: Ayush Kapoor
"""

#%% Importing the necessary Library
from os import path
import sys
import copy
import time
import matplotlib as plot
import seaborn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_precision_recall_curve

#%% Function Declaration

def timeElapsed (start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
def get_fig(dataSet, x_axis, y_axis, hue_):
    pt= seaborn.lmplot(x=x_axis, y=y_axis, data=dataSet, hue= hue_, fit_reg=False)
    #figure = pt.fig
    #figure.savefile ("Figure1.png")
    
def drawPRCurve(classifier_ , attribute, y, y_, extraLabel):
    disp = plot_precision_recall_curve(classifier_, attribute, y)
    disp.ax_.set_title('Binary Classification Precision-Recall curve: '+extraLabel)

def execute_kNN (trainData, validateData, testData,k_value):
    # We are here taking the value of K i.e., we are here telling the
    # classfier that we need to consider 4 nearest neighbour 
    classifier = KNeighborsClassifier(n_neighbors= k_value)
    classifier.fit(trainData.iloc[:, :len(trainData.columns)-1], trainData.iloc[:,len(trainData.columns)-1])    
    
    validate_pred = classifier.predict(validateData.iloc[:,:len(validateData.columns)-1])
    print ("INFO: k-NN Validation Results follows")
    knnCM1= confusion_matrix(validateData.iloc[:,len(validateData.columns)-1], validate_pred)
    print(knnCM1)
    print("RESULT: Validation Accuray is ",((knnCM1[0][0] + knnCM1[1][1]) / (knnCM1.sum())) *100)
    
    test_pred = classifier.predict(testData.iloc[:,:len(testData.columns)-1])
    print ("INFO: k-NN Test data Results follows")
    knnCM2 = confusion_matrix(testData.iloc[:,len(testData.columns)-1], test_pred)
    print(knnCM2)
    print("RESULT: k-NN Test Accuray is ",((knnCM2[0][0] + knnCM2[1][1]) / (knnCM2.sum())) *100)
    # Drawing a Presion Recall Curve
    drawPRCurve(classifier , testData.iloc[:,:len(testData.columns)-1],testData.iloc[:,len(testData.columns)-1], test_pred, " for kNN")
    return ((knnCM2[0][0] + knnCM2[1][1]) / (knnCM2.sum())) *100 # Returning the accuracy of the prediction 

def execute_decisionTree(trainData, validateData, testData, dt_criterion):
    decisionTree = DecisionTreeClassifier(random_state = 0, criterion = dt_criterion)
    # Training process
    decisionTree.fit(trainData.iloc[:, :len(trainData.columns)-1], trainData.iloc[:,len(trainData.columns)-1])
    print ("INFO: Decision Tree Validation Results follows")
    # Checking the validity of the model by using validation data set
    validate_pred = decisionTree.predict(validateData.iloc[:,:len(validateData.columns)-1])
    dtCM1= confusion_matrix(validateData.iloc[:,len(validateData.columns)-1], validate_pred)
    print(dtCM1)
    print("RESULT: Decion Tree Validation Accuray is ",((dtCM1[0][0] + dtCM1[1][1]) / (dtCM1.sum())) *100)
    print("INFO: Decision Tree Test Results follows")
    # Finally testing the model on the unseen data set
    test_pred = decisionTree.predict(testData.iloc[:,:len(testData.columns)-1])
    dtCM2= confusion_matrix(testData.iloc[:,len(testData.columns)-1], test_pred)
    print(dtCM2)
    print("RESULT: Decion Tree Test Accuray is ",((dtCM2[0][0] + dtCM2[1][1]) / (dtCM2.sum())) *100)
    drawPRCurve(decisionTree , testData.iloc[:,:len(testData.columns)-1],testData.iloc[:,len(testData.columns)-1], test_pred, " for Decision Tree")
    return ((dtCM2[0][0] + dtCM2[1][1]) / (dtCM2.sum())) *100
    
def execute_randomForest(trainData, validateData, testData, kernel_):
    svc_classifier = SVC(random_state = 0, kernel = kernel_) # rbf
    svc_classifier.fit(trainData.iloc[:, :len(trainData.columns)-1], trainData.iloc[:,len(trainData.columns)-1])
    # Checking the validity of the Random Forest
    print ("INFO: Random Forest Validation Results follows:")
    validate_rf = svc_classifier.predict(validateData.iloc[:,:len(validateData.columns)-1])
    rfCM1 = confusion_matrix(validateData.iloc[:,len(validateData.columns)-1], validate_rf) 
    print(rfCM1)
    print("RESULT: Random Forest Validation Accuray is ",((rfCM1[0][0] + rfCM1[1][1]) / (rfCM1.sum())) *100,"\n")
    print("INFO: Random Forest Test Results follows:")
    test_rf = svc_classifier.predict(testData.iloc[:,:len(testData.columns)-1])
    rfCM2 = confusion_matrix(testData.iloc[:,len(testData.columns)-1], test_rf) 
    print(rfCM2)
    print("RESULT: Random Forest Test Accuray is ",((rfCM2[0][0] + rfCM2[1][1]) / (rfCM2.sum())) *100,"\n")
    drawPRCurve(svc_classifier , testData.iloc[:,:len(testData.columns)-1],testData.iloc[:,len(testData.columns)-1], test_rf, " for Random Forest")
    return ((rfCM2[0][0] + rfCM2[1][1]) / (rfCM2.sum())) *100  
           
#%% Variable/Parameter Declaration
# These are used in the algorithm and other procedure and can be changed, if required
train_ratio = 70
validate_ratio = 15
test_ratio = 15
k_value = 4
imp_strategy = 'median'
dt_criterion = 'entropy'
rf_kernel = 'rbf'

#%% Section 1: User Input 
dataset_file = input ("Kindly input the complete path of the .xls/.csv data file --> \n")
if str(path.exists(dataset_file)) == "False":
    print ("ERROR: File Not Found")
    sys.exit ()
print ("INFO: Dataset Found Successfully.\n")

#%% Section 2: Reading and extracting data
# Check whether the file is an .xls or a .csv and then process
# it accordingly
if (dataset_file.split('/')[-1].split('.')[-1] == "csv"):
    dataset = pd.read_csv(dataset_file)
if (dataset_file.split('/')[-1].split('.')[-1] == "xls"):
    dataset = pd.read_excel(dataset_file)
dataset_backup = copy.deepcopy(dataset)    
    
#%% Section 3: Data Processing 
get_fig(dataset_backup, 'Time', 'Amount', 'Class')   
print("DEBUG: Attributes Shape:", dataset.iloc[: , 1:len(dataset.columns)-1].values.shape)
print("DEBUG: Output Shape:", dataset.iloc[:,len(dataset.columns)-1].values.shape,"\n")    

# Filling the missing values in the dataset
print("INFO: ",np.isnan(dataset)[np.isnan(dataset) == False].size,"Empty entries found")
imp = SimpleImputer(missing_values= np.nan, strategy= imp_strategy)
imp = imp.fit(dataset.iloc[:, 1:len(dataset.columns)-1])
dataset.iloc[:, 1:len(dataset.columns)-1] = imp.fit_transform(dataset.iloc[:, 1:len(dataset.columns)-1])    
print("INFO: Data Imputation Over")

# Scaling data for uniformity
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(dataset.iloc[:, 1:len(dataset.columns)-1])
dataset.iloc[:, 1:len(dataset.columns)-1] = scaler.transform(dataset.iloc[:, 1:len(dataset.columns)-1])
print("INFO: Data Scaling Over\n")

# Distributing training, Validation & Testing
fraud_case = shuffle(dataset[dataset["Class"]==1])
non_fraud_case = shuffle(dataset[dataset["Class"]==0])
temp_fraud = fraud_case.iloc[:int(np.floor((len(fraud_case)*train_ratio)/100)),1:]
temp_nfraud = non_fraud_case.iloc[:int(np.floor((len(non_fraud_case)*train_ratio)/100)),1:]
# Final training Data Set
trainData = shuffle(pd.concat([temp_fraud,temp_nfraud])) 
print ("DEBUG: Training Data Shape:",trainData.shape)

temp_fraud = fraud_case.iloc[int(np.floor((len(fraud_case)*train_ratio)/100)):int(np.floor((len(fraud_case)*train_ratio)/100))+int(np.floor((len(fraud_case)*validate_ratio)/100)),1:]
temp_nfraud = non_fraud_case.iloc[int(np.floor((len(non_fraud_case)*train_ratio)/100)):int(np.floor((len(non_fraud_case)*train_ratio)/100))+int(np.floor((len(non_fraud_case)*validate_ratio)/100)),1:]
# Final Validation Data Set
validateData = shuffle(pd.concat([temp_fraud,temp_nfraud]))
print ("DEBUG: Validation Data Shape:",validateData.shape)

temp_fraud = fraud_case.iloc[int(np.floor((len(fraud_case)*train_ratio)/100))+int(np.floor((len(fraud_case)*validate_ratio)/100)):,1:]
temp_nfraud = non_fraud_case.iloc[int(np.floor((len(non_fraud_case)*train_ratio)/100))+int(np.floor((len(non_fraud_case)*validate_ratio)/100)):,1:]
# Final Test Data Set
testData = shuffle(pd.concat([temp_fraud,temp_nfraud]))
print ("DEBUG: Testing Data Shape:",testData.shape)
print("INFO: Data Distribution over\n")

#%% Section 4: Algorithm Application    

# Computing by using k-NN Approach
print("INFO: k-NN Excution Started. Kindly wait")
start = time.time()
knnAccuracy= execute_kNN (trainData, validateData, testData,k_value)
end = time.time()
print ("INFO:Time Elapsed for k-NN is ", timeElapsed(start,end))

# Computing using Decision Tree Approach 
print("INFO: Decision Tree Excution Started. Kindly wait")
start = time.time()   
dtAccuracy = execute_decisionTree(trainData, validateData, testData, dt_criterion)    
end = time.time()
print ("INFO:Time Elapsed for Decision Tree is ", timeElapsed(start,end))    

# Computing using Random Forest Approach
print("INFO: Random Forest Excution Started. Kindly wait")
start = time.time()     
rfAccuracy = execute_randomForest(trainData, validateData, testData, rf_kernel)   
end = time.time()
print ("INFO:Time Elapsed for Random Forest is ", timeElapsed(start,end))      

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

