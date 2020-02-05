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

#Path to folder with output of CoreML
path="./UniversalBank"

#Use auxiliar function "deploy_helper"
# this will configure the objects
# to exactly reproduce the training set-up
# in new data.
predictor=deploy_helper(path)

#Read data
path_csv="./Datasets/UniversalBank.csv"
data_t=pd.read_csv(path_csv, delimiter=";",decimal=".")

#This is an example with the same data used for training.
# removing the target is necessary. On production the target
# class is not available.
y_real=data_t["Target"]
data_t.drop("Target",axis=1,inplace=True)
#Create DataFrame of predictions.
predictions=predictor.make_predictions(data_t,False,)

print(predictions)
