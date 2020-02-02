"""This script is an example
on how to use CoreML, which principal
functions is called core_diagnosis, and
output is the score on test set.

Note: CoreML will write objects for
future deployment plus a report
on curret directory under the name
"name" (input of core_diagnosis).
"""

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
