import os
import sys
# Absolute path of Predictor.py
script_pos = os.path.dirname(__file__)

# Include Classes in the current python enviroment
if not script_pos in sys.path:
    sys.path.append(script_pos)

#Static class methods for machine learning pre
# pre processing.
from StaticML import StaticML
#General functions for data pre prosessing
from DataStandardizer import DataStandardizer
#DataFrame library
import pandas as pd


class Predictor():
    """Class to use best model output from AML_MS
    .Use this class to deploy trainied classifiers.
    """

    def __init__(self,best_clf,min_max_scaler,
                clustering_obj,best_features,
                best_features_poly,poly_obj,encoders):
        """Use Pickle files from CoreML to deploy
        the classifier.
        Parameters:
            best_clf (object): classifier from CoreML
            min_max_scaler (object)
            clustering_obj (object)
            best_features (list)
            best_features_poly (list)
            poly_obj (object)
        """
        self.best_clf=best_clf
        self.min_max_scaler=min_max_scaler
        self.clustering=False
        self.polynomial=False

        #Set objects and variables according to 
        # user input.
        if clustering_obj is not None:
            self.clustering_obj=clustering_obj
            self.clustering=True
        if best_features is not None:
            self.best_features=best_features
        if best_features_poly is not None:
            self.best_features_poly = best_features_poly
        if poly_obj is not None:
            self.poly_obj=poly_obj
            self.polynomial=True
        if encoders is not None:
            self.encoders=encoders


    def prepare_for_predict(self,X):
        """pipeline of transformations to make a prediction."""

        X = StaticML.transform_min_max(X,self.min_max_scaler)

        if self.clustering:
            X=StaticML.transform_clustering(X,
                                        self.clustering_obj)
        if self.polynomial:
            X=StaticML.transform_polinomial_df(X[self.best_features],self.poly_obj)
            X=X[self.best_features_poly]
        #Join categorical columns if they exist.
        if(len(self.df_cate)!=0):
            X=X.join(self.df_cate)

        return X


    def make_predictions(self,df,probability):
        """make predictions on new data.
        This data must be formatted exacly the 
        same as the onw used for training.
        Paramaters:
            df (DataFrame): new data.
            probability (boolean): to return
                probabilities or raw predictions.
        returs:
            df_prediction (DataFrame): probability
                of class.
        """
        Ds=DataStandardizer()
        #Pass only features.
        df_vals,df_cate=Ds.pipe_line_stand(df,mode="no_target")

        i=0
        self.df_cate=df_cate
        if not df_cate.empty:
            for each_col in list(df_cate.columns):
                df_temp=df_cate[[each_col]]
                if(i==0):
                    df_encoded=Ds.transform_encoded_single_df(df_temp,
                                                self.encoders.get(each_col))[0]
                else:
                    temp=Ds.transform_encoded_single_df(df_temp,
                                                self.encoders.get(each_col))[0]
                    df_encoded=df_encoded.join(temp,lsuffix=str(i))
                i=i+1
            self.df_cate=df_encoded

        self.X=self.prepare_for_predict(df_vals)

        if probability:
            prediction=self.best_clf.predict_proba(self.X)[:,1]
        else:
            prediction=self.best_clf.predict(self.X)

        df_prediction=pd.DataFrame(prediction,
                                    index=self.X.index,
                                    columns=["Predicted_Target"])

        return df_prediction





