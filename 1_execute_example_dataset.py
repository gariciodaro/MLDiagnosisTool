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
score_1=CoreML.core_diagnosis(data_origin="/home/gari/Desktop/folders/betting/data/curated_data_set_tournament.file",
                  data_origin_string="pickle_df",
                  name="/tennis",
                  string_metric="roc_auc_score",
                  number_of_clusters=40,
                  hiper_tunning_iters=30,
                  futher_op_boo=True,
                  balance_train=False,
                  clustering=True,
                  polynomial=True,
                  test_size=0.3)
print("tennis",score_1) 
