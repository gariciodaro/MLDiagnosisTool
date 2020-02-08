


<img src="http://garisplace.com/img/default-monochrome-black.svg" width="50%"/>

---

<center> <h2> Automatic Machine Learning Diagnosis Tool </h2> </center>

I developed a set of python scripts and classes that take as input 
a Data Frame or a .csv and perform common machine learning techniques,
both pre-processing and processing. After execution, you will get a new folder
containing an HTML report( [see](http://garisplace.com/UniversalBank.html) here an example) plus all the required pickle files
to deploy the best model over your data set. This should give you a guide
of what path to follow on your search space of optimization.


<center> 
<img src="http://garisplace.com/img/dig_flow.png" width="50%"/>
</center>

<h3> Installation: </h3> 

Tested using anaconda.
On terminal (respect the installation order!):
1. `conda create -n MLDiagnossisTool python=3.6`
2. `conda activate MLDiagnossisTool`
	1. --on ubuntu: `conda install -c conda-forge xgboost==0.90`
	1. --on Windows `conda install -c anaconda py-xgboost==0.90`
3. `conda install scikit-learn=0.21.3`
4. `conda install pandas=0.25.3`
5. `conda install matplotlib=3.1.0`
6. `conda install seaborn==0.9.0`
7. `conda install -c districtdatalabs yellowbrick==1.0.1`
8. `conda install -c conda-forge imbalanced-learn==0.5.0`
9. `conda install -c anaconda jinja2==2.10.3`


<h3> Example of use: </h3> 

Run on python environment:
`python 1_execute_example_dataset.py`

``` python
import CoreML
import pandas as pd

# DO NOT FORGET THE FORWARD SLASH "/" ON name
# name="/UniversalBank",
df_1=pd.read_csv("./Datasets/UniversalBank.csv", delimiter=";",decimal=".")
score_1=CoreML.core_diagnosis(data_origin=df_1,
                  data_origin_string="object_dataframe",
                  name="/UniversalBank",
                  string_metric="roc_auc_score",
                  number_of_clusters=40,
                  hiper_tunning_iters=15,
                  futher_op_boo=True,
                  balance_train=False,
                  clustering=True,
                  polynomial=True,
                  test_size=0.3)
print("UniversalBank",score_1)
```

After the prior execution, you can deploy on new data (this is just an example,
so I will use the same training data.)

Run on python environment:
`python 2_execute_production_example.py`

``` python
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
predictions=predictor.make_predictions(data_t,False)

print(predictions)
```

<h3> Comments: </h3> 

+ The software is prepared to read dataframe or csv, however, the csv must be separated by semi columns “;”. 
+ The target variable must be the last column.
+ The software can only be used for binary classification.
+ It is able to handle categorical data. 
+ It is able to handle missing values, on target removes the register, on numerical replace by the mean, on categorical also remove (This feature was deactivated, user must pass data set without missing data). 
+ If binary data is included in the features, they cannot be integers. For example, Gender cannot be 1 and 2, it must be “male” or “female”. This is due to the automatic detection of type data, in the case of Gender =1 or 2, the feature will be treated as numeric instead of categorical and won’t be hot encoded as intended.
+ The file 1_execute_example_dataset.py, contains the recommended parameters for core_diagnosis function. This configuration is balanced in terms of computational time and final score. Increasing the parameter hiper_tunning_iters is the most reliable way of increasing the score, but the computational time will also increase accordingly.
+ From my point of view, this software is a guiding tool rather than a final ruler. That is why I programmed it with so many parameters, and why the report is so important. For example, only a human can decide that in a giving situation, Quadratic Discriminant Analysis that scored 89% is a better option than Multi layer perceptron that score 95%, because the speed of prediction is QDA is much faster than MLP.
+ UniversalBank is data set randomly extracted from Kaggle.
+ folders data_banknote_authentication and UniversalBank are examples of outputs of execution.