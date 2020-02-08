"""This script is an example
on how to use CoreML, which principal
functions is called core_diagnosis, and
output is the score on test set.

Note: CoreML will write objects for
future deployment plus a report
on curret directory under the name
"name" (input of core_diagnosis).

Extracted from CoreML.py
data_origin (object or string): posible values
                              "object_dataframe":
                              "csv"
                              "pickle_df"

"""

import CoreML
import pandas as pd

# DO NOT FORGET THE FORWARD SLASH "/" ON name
# name="/data_banknote_authentication",
path_csv="./Datasets/data_banknote_authentication.csv"
data_t=pd.read_csv(path_csv, delimiter=",",decimal=".")

score_1=CoreML.core_diagnosis(data_origin=data_t,
                  data_origin_string="object_dataframe",
                  name="/data_banknote_authentication",
                  string_metric="accuracy_score",
                  number_of_clusters=30,
                  hiper_tunning_iters=15,
                  futher_op_boo=True,
                  balance_train=False,
                  clustering=True,
                  polynomial=True,
                  test_size=0.3)
print("tennis",score_1) 