#This class applies required transformations
# to use classifiers form CoreML
from Classes.Predictor import Predictor
#Interaction between filesystem and python Environment.
#Uses pickle.
from OFHandlers import OFHandlers as OFH
import os

def deploy_helper(path):
    """use clasifier
    Paramters:
        path (string): folder of pickle file
        output of CoreML
    """

    #Read all pickle files.
    file_names=[file for file in os.listdir(path) if 
                file.endswith(".file")]
    #Initialized variables for class Predictor
    clustering_obj=None
    best_features_poly=None
    best_features=None
    poly_obj=None
    encoders=None

    #Check and assign variables for class Predictor
    for each_file_name in file_names:
        each_file_name_s=each_file_name.split("_")
        if ("clf.file" in each_file_name_s):
            best_clf=OFH.load_object(path+"/"+each_file_name)

        elif ("scaler.file" in each_file_name_s):
            min_max_scaler=OFH.load_object(path+"/"+each_file_name)

        elif ("clustering" in each_file_name_s):
            clustering_obj=OFH.load_object(path+"/"+each_file_name)

        elif ("features" in each_file_name_s) and ("poly.file" in each_file_name_s):
            best_features_poly=OFH.load_object(path+"/"+each_file_name)

        elif ("features.file" in each_file_name_s):
            best_features=OFH.load_object(path+"/"+each_file_name)

        elif ("poly" in each_file_name_s) and ("obj.file" in each_file_name_s):
            poly_obj=OFH.load_object(path+"/"+each_file_name)

        elif ("catencoders.file" in each_file_name_s):
            encoders=OFH.load_object(path+"/"+each_file_name)

    predictor=Predictor(best_clf=best_clf,
                        min_max_scaler=min_max_scaler,
                        clustering_obj=clustering_obj,
                        best_features=best_features,
                        best_features_poly=best_features_poly,
                        poly_obj=poly_obj,
                        encoders=encoders)
    return predictor



