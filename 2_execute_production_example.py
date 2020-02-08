"""This script is an example
on how to use the outputs of CoreML.
This will return a dataFrame of
predictions.
"""

#Auxiliar function that will read a folder
# that contains the pickle files output of
# CoreML.
from Deploy import deploy_helper
import pandas as pd


"""
from sklearn.metrics import (roc_auc_score,
                             recall_score,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             roc_curve)
"""


#Path to folder with output of CoreML
path="./data_banknote_authentication"

#Use auxiliar function "deploy_helper"
# this will configure the objects
# to exactly reproduce the training set-up
# in new data.
predictor=deploy_helper(path)

#Read data
path_csv="./Datasets/data_banknote_authentication.csv"
data_t=pd.read_csv(path_csv, delimiter=",",decimal=".")

#This is an example with the same data used for training.
# removing the target is necessary. On production the target
# class is not available.
y_real=data_t["Target"]
data_t.drop("Target",axis=1,inplace=True)
#Create DataFrame of predictions.
predictions=predictor.make_predictions(data_t,False)

# to double check. This number should be the same as 
# the bullet point in the repor "Training score for full data set"
#print(accuracy_score(y_real, predictions))

print(predictions)
