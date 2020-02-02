#Linear algebra
import numpy as np
#Machine learning classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
#Machine learning metrics
from sklearn.metrics import (roc_auc_score,
                             recall_score,
                             f1_score,
                             average_precision_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             roc_curve)


class StaticDictML:
    """Dictionaries that are used in class
    ModelingProcedure. Be sure to update all methods
    when a new model is introduced.
    """

    
    def __init__(self):
        pass

    @staticmethod
    def get_dict_models():
        dict_models={"RF":RandomForestClassifier,
                    "LgR":LogisticRegression,
                    "XGB":XGBClassifier,
                    "MLP":MLPClassifier,
                    "LDA":LinearDiscriminantAnalysis,
                    "QDA":QuadraticDiscriminantAnalysis,
                    "KNN":KNeighborsClassifier
                    }
        return dict_models

    @staticmethod
    def get_dict_metrics():
        dict_metrics={"roc_auc_score":roc_auc_score,
                      "recall_score":recall_score,
                      "f1_score":f1_score,
                      "average_precision_score":average_precision_score,
                      "accuracy_score":accuracy_score,
                      "balanced_accuracy_score":balanced_accuracy_score}
        return dict_metrics

    @staticmethod
    def get_roc_curve(y_actual,y_predicted):
        fpr, tpr, _ = roc_curve(y_actual, y_predicted)
        auc_score=roc_auc_score(y_actual, y_predicted)
        return fpr,tpr,auc_score


    @staticmethod
    def get_parameter_grid(string_model):
        grid=None
        if(string_model=='RF'):
            # Number of trees in random forest
            n_estimators = [int(x) for
                             x in np.linspace(start = 1, 
                                            stop = 100, 
                                            num = 30)]
            n_estimators.append("warn")
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]# Create the random grid
            grid = {'n_estimators': n_estimators,
            'random_state':[0,10,42,13],
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}

        if(string_model=='KNN'):
            k_range = list(range(1,31))
            weight_options = ["uniform", "distance"]
            grid = dict(n_neighbors = k_range, 
                        weights = weight_options)

        if(string_model=='MLP'):
            grid = {'activation' : ['logistic'],
                'max_iter':[7000],
                'random_state':[0,10,42,13],
                'solver' : ['adam'],
                'alpha': [0.0001],
                'hidden_layer_sizes': [(100),(100,100),
                                        (100,100,100),
                                     (100,100,100,100),
                                     (100,100,100,100,100),
                                     (145),(150),(160),
                                     (170),(50)]
            }

        if(string_model=='XGB'):
            grid = {"min_child_weight":[1],
            "max_depth":[3,7,9,10],
            "learning_rate":[0.08,0.09,0.1,0.3,0.4],
            "reg_alpha":[0.01,0.02,0.3],
            "reg_lambda":[0.8,0.9,1,1.2,1.4],
            "gamma":[0.002,0.004,0.03],
            "subsample":[0.9,1.0],
            "colsample_bytree":[0.7,0.9,1.0],
            "objective":['binary:logistic'],
            "nthread":[-1],
            "scale_pos_weight":[1],
            "seed":[0,10,42,13],
            "n_estimators": [50,100,200,300,400,500]}

        if(string_model=='LgR'):
            grid = {"penalty":["l2"],
            "C":[0.1,0.2,0.3,0.5,1.0,1.5],
            "class_weight":[None,"balanced"],
            "random_state":[0,10,42,13],
            "solver":["lbfgs","sag"]}

        if(string_model=='LDA'):
            grid = {"solver":["lsqr"],
            "shrinkage":[None,"auto",0.0,0.83,
                        0.01,0.02,0.03,0.2,0.5,0.4,0.7,
                        0.015,0.026,0.344,0.04,0.00001,
                        0.45,0.55]}
        if(string_model=='QDA'):
            grid = {"reg_param":[0.0,0.83,0.01,0.02,
                                0.03,0.2,0.5,0.4,0.7,
                                0.015,0.026,0.344,0.04,
                                0.00001,0.45,0.55]}
        return grid