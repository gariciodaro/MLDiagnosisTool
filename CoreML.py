#To measure the time duration of one execution
import time
#Library for dataframe creation
import pandas as pd

#Linear algebra
import numpy as np

"""System-specific parameters and functions
This module provides access to some variables used or
maintained by the interpreter and to functions 
that interact strongly with the interpreter.
"""
import sys

"""Miscellaneous operating system interfaces
This module provides a portable way of using 
operating system dependent functionality
"""
import os

# Absolute path of CoreML.py script
script_pos = os.path.dirname(__file__)

# Absolute path of the classes needed to 
# run CoreML.py script
classes_pos = script_pos+"/Classes"

# Include script_pos in the current python enviroment
if not script_pos in sys.path:
    sys.path.append(script_pos)

# Include classes_pos in the current python enviroment
if not classes_pos in sys.path:
    sys.path.append(classes_pos)

#Since classes_pos was included in the current Environment
#we can import our classes directly:

#Interaction between filesystem and python Environment.
#Uses pickle.
from OFHandlers import OFHandlers as OFH

#General functions for data pre prosessing
from DataStandardizer import DataStandardizer

#Core machine learning pipeline
from ModelingProcedure import MachineLearning

#Auxliar class for plotting the core machine learning
from PlotterForReport import PlotterForReport 

#Main component of the html report.It is used
#for html template manipulation from python.
from jinja2 import Environment,FileSystemLoader

#Auxiliar dictionary operations.
from DictionaryAuxiliar import DictionaryAuxiliar



def core_diagnosis(data_origin_string,
           data_origin,
           name=None,
           string_metric="roc_auc_score",
           number_of_clusters=35,
           hiper_tunning_iters=10,
           futher_op_boo=True,
           balance_train=True,
           clustering=True,
           polynomial=True,
           test_size=0.1):
    """Core function of the 
        automatic diagnosis tool for machine learning software

    Parameters:
        data_origin_string (string): Type of data input. posible values:
            object_dataframe
            csv
            pickle_df

        data_origin (object or string): Depending on data_origin_string 
            data frame object
            path to csv file
            pah to pickle object

        name (string): name to be used as path report folder 

        string_metric: (string): machine learning metrics to be 
            optimized, posible values:
            roc_auc_score
            recall_score
            f1_score
            average_precision_score
            accuracy_score
            balanced_accuracy_score

        number_of_clusters (int): maximum number of clusters to 
            calculate performing the 
            elbow optimization method
            for k means clustering.

        hiper_tunning_iters (int): maximum number of iterations when 
            random searching for best 
            set of hyperparameter

        futher_op_boo (boolean): Whether to perform hyperparameter 
            optimization around the best 
            hyperparameters found while
            performing hiper_tunning_iters.
            Warning: While testing it was observed
            that this will increase the metric 
            around 2% at a huge computational cost.
            Use it when appropriate.

        balance_train (boolean): Whether to balance the dataset target
            classes. Random oversampling 
            was the technique implemented.
            Warning: Choose the metric accordingly,
            if for example accuracy_score is used
            on an imbalanced dataset, the naive 
            approach might be the one selected, 
            this means a model whose predictions 
            are always the most common class
            will outperform other more 
            useful models.

        clustering (boolean): Whether to create and test clustering 
            dimension reduction 

        polynomial (boolean): Whether to create and test polynomial 
            features.
            Warning: if  [a, b] is a set of features,
            the degree-2 polynomial features will be 
            [a, b, a^2, ab, b^2], use carefully, 
            since the synthetic features can easily 
            overflow the memory. Internally a selection 
            of 100 best features is performed using 
            the chi statistic test, but even if only 100 
            are selected, at some point, all are computed.

        test_size (float): size from [0.1, 0.9] of test set.
            Note: This number will be used to test 
            the performance of the last tunned model, 
            this portion of the data is not used on 
            cross-validation.


     Returns:
        score_on_test (float) : Score on test set.
        Writes on the folder ./name:
        name_clf.file: A pickle file of the best classifier.
        name_scaler.file: A pickle file of scale object.
        name_clustering_obj.file: pickle file of the clustering object
        name_best_features.file: pickle file of a the string set
            of best columns on chi test of
            before aplying polinomial generation.
        name_best_features_poly.file: pickle file of a the string set 
                of best columns on chi tesr after 
                name_best_features selection.
        name_poly_obj.file: pickle file of the polynomial object.
        name.html: an html file with the report of the execution.
            here you find base line scores, metrics, iterations
            and general comments of the execution.

        Note: name refers to the paramater "name" that is input to the
            function core_diagnosis.

    """

    if(number_of_clusters<25):
        msg1="a number of cluster smaller than 25 is not allowed"
        msg2=", setting number_of_clusters to 35"
        print(msg1+msg2)
        number_of_clusters=35

    # start measuring the duration of execution.
    start_time = time.time()

    #Load the data using the input parameters.
    if data_origin_string=="object_dataframe":
        df_1=data_origin
    elif data_origin_string=="csv":
        df_1=pd.read_csv(data_origin, delimiter=";",decimal=".")
    elif data_origin_string=="pickle_df":
        df_1=OFH.load_object(data_origin)

    #If name is not set, a default folder
    # is created, where all the output
    # will be written.
    if name is None:
        name="/current_project"
    #Try to create containing folders,
    # if they already exists pass.
    try:
        os.mkdir(script_pos+name)
        os.mkdir(script_pos+name+"/img")
    except:
        pass
    #Set path for output.
    working_directory=script_pos+name

    #Instate DataStandardizer
    Ds=DataStandardizer()

    #Pre process the dataset.
    df_1,df_cate,target_1=Ds.pipe_line_stand(df_1)
    #print("categorical_columns",df_cate.columns)
    #print("null cols in df_1",DataStandardizer.get_null_coll(df_1))
    #print("null cols in df_cate",DataStandardizer.get_null_coll(df_cate))

    #Instate MachineLearning class
    # the core machine learning pipelines.
    MP_1=MachineLearning(df=df_1,
                        df_cate=df_cate,
                        target=target_1,
                        string_metric=string_metric,
                        balance_train=balance_train,
                        test_size=test_size)

    # test with default parameters
    # on three available modes:
    #   Raw data
    #...Dimension reduced by clustering.
    #...polynomial features.
    #
    # Note that the available modes are visited in agreement to 
    # the boolean inputs clustering and polynomial.
    # On the best mode, it selects the best model using
    # the highest cross-validation error, and model with
    # with smallest standard deviation for further optimization
    # (random search of set of hyperparameters).
    # It also saves the corresponding plots on the ./name/img 
    # folder.
    #
    # Note 2: clustering and polynomial booleans get updated 
    # and are not longer the ones set by the user.
    (str_model_best_score,
        str_model_min_std,
        clustering,
        polynomial)=MP_1.configurations_and_plot(number_of_clusters,
                                                 clustering,
                                                 polynomial,
                                                 working_directory)

    #print("Best base model on cv score ",str_model_best_score)
    #print("Best base model on min std ",str_model_min_std)
    print("Starting hiper parameters tunning...")

    # Random search of hyperparameters on best model based
    # on cross validation error
    cvb_best_score_tunning=MP_1.hiper_tunning(str_model_best_score,
                                                clustering=clustering,
                                                polynomial=polynomial,
                                                max_evals=hiper_tunning_iters)

    # Random search of hyperparameters on best model based
    # on smallest standard deviation.
    cvb_best_std_tunning=MP_1.hiper_tunning(str_model_min_std,
                                                clustering=clustering,
                                                polynomial=polynomial,
                                                max_evals=hiper_tunning_iters)

    msg3="hiper tunning results on "
    print(msg3+ str_model_best_score+ " \n",cvb_best_score_tunning)
    print(msg3+ str_model_min_std+ " \n",cvb_best_std_tunning)

    # This method decides based con cv score which
    # which of the two candidates models
    # is best, and in agreement to the boolean input "futher_op_boo"
    # performs a further optimization of hyperparameters
    # by moving in a sequential manner around the currently best
    # hyperparameters. It is also required that cv score is smaller than 0.91
    # for further optimization to occur.
    MP_1.further_optimize_pipeline(cvb_best_score_tunning=cvb_best_score_tunning,
                                   cvb_best_std_tunning=cvb_best_std_tunning,
                                   futher_op_boo=futher_op_boo,
                                   clustering=clustering,
                                   polynomial=polynomial)

    #Fit the final classifier and save all the pickle files
    # that are necessary to run the classifier on production.
    MP_1.build_final_clf(clustering,polynomial,working_directory,name)

    # Saves the actual best classifier to be used in production.
    OFH.save_object(working_directory+name+"_clf.file",MP_1.trained_model)

    #Using the jinja2 to create the report on html
    # see the folder ./Report/report.html if you would
    # like to modify the layout of the report.
    file_loader = FileSystemLoader(script_pos+"/Report")
    env = Environment(loader=file_loader)
    template = env.get_template("report.html")
    output = template.render(title=name,data_set_name=name,
            results1=cvb_best_score_tunning,
            results2=cvb_best_std_tunning,
            clustering=clustering,
            polynomial=polynomial,
            balance_train=balance_train,
            selected_model=MP_1.string_model,
            super_parameters=MP_1.super_parameters,
            final_training_score=MP_1.final_training_score,
            trained_score=MP_1.trained_scored,
            futher_op_boo=futher_op_boo,
            improved_score_further_op=MP_1.improved_score_further_op,
            expected_score=MP_1.expected_score,
            optimized_metric=string_metric,
            processing_time=time.time() - start_time,
            working_directory=working_directory)

    print("selected_model:",MP_1.string_model)
    print("selected parameters:",MP_1.super_parameters)
    print("score on test set of data:",MP_1.expected_score)
    print("Do not forget to see your report in:", working_directory)

    # Write the actual html of the report on folder.
    with open(working_directory+name+".html", 'w') as f:
        f.write(output)

    print("--- %s seconds ---" % (time.time() - start_time))
    score_on_test=MP_1.expected_score
    return score_on_test
    #return working_directory+"/"+name


