#DataFrame library
import pandas as pd
# Synthetic polynomial features
from sklearn.preprocessing import PolynomialFeatures
# Selection of features on chi test.
from sklearn.feature_selection import SelectKBest, chi2
# Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
# Technique for balance data.
from imblearn.over_sampling import RandomOverSampler


class StaticML:
    """Static methods auxiliar to machine learning.
    Method required both for training and test.
    """


    def __init__(self):
        pass

    @staticmethod
    def transform_clustering(X,clustering_obj):
        """
        Parameters:
            X (DataFrame): Features
            clustering_obj (object): k means clustering object.
        Returns:
            cluster_l (DataFrame): Data transformed into
                cluster distance space
        """
        elbow=clustering_obj.n_clusters
        cluster_labels=["d_"+str(each) for each in range(1,elbow+1)]
        cluster_l=pd.DataFrame(clustering_obj.transform(X),
                            index=X.index,
                            columns=cluster_labels)
        return cluster_l

    @staticmethod
    def get_polinomial_df(df_X,degree=2):
        """Parameters:
            df_X (DataFrame): Features
            degree (int).
        Returns:
            df_X_poly (DataFrame): Data transformed
                into polynomial space.
            poly_object (object)
        """
        poly_object = PolynomialFeatures(degree,include_bias=False)
        X_poly=poly_object.fit_transform(df_X)
        cols_poly=poly_object.get_feature_names(df_X.columns)
        df_X_poly=pd.DataFrame(X_poly,columns=cols_poly,index=df_X.index)
        return poly_object,df_X_poly

    @staticmethod
    def transform_polinomial_df(df_X,poly_object):
        """
        Parameters:
            df_X (DataFrame): Features
            poly_object (object): Polynomial object.
        Returns:
            df_X_poly (DataFrame): Data transformed into
                polynomial space.
        """
        X_poly=poly_object.transform(df_X)
        cols_poly=poly_object.get_feature_names(df_X.columns)
        df_X_poly=pd.DataFrame(X_poly,columns=cols_poly,index=df_X.index)
        return df_X_poly

    @staticmethod
    def select_k_best(df_X,df_y,k):
        """Select a reduced number of features
            based on chi statistic test.
        Parameters:
            df_X (DataFrame): Features
            df_y (DataFrame): Target.
            k (int): recommended maximum 
                number of features.
        Returns:
            selected_cols (list): selected
                columns based on chi test.
        """
        features_cols=df_X.columns

        #Try the recommended maximun, if not,
        # select all features.
        try:
            k_best = SelectKBest(chi2, k=k)
            X_best=k_best.fit_transform(df_X, df_y)
        except:
            k_best = SelectKBest(chi2, k="all")
            X_best=k_best.fit_transform(df_X, df_y)
        features_cols_index=k_best.get_support(indices=True)
        selected_cols=[features_cols[each_col] for 
                            each_col in features_cols_index]
        return selected_cols

    @staticmethod
    def min_max_df(df_X):
        """ Min max scaler.
        Parameters:
            df_X (DataFrame): Features.
        Returns:
            scaler (object)
            X_scaled (DataFrame)
        """
        scaler = MinMaxScaler()
        X_scaled=scaler.fit_transform(df_X)
        X_scaled=pd.DataFrame(X_scaled,
            columns=df_X.columns,index=df_X.index)
        return scaler,X_scaled

    @staticmethod
    def transform_min_max(df_X,min_max_scaler):
        """
        Parameters:
            df_X (DataFrame): Features
            min_max_scaler (object): min_max_scaler object.
        Returns:
            X_scaled (DataFrame): Data scaled.
        """
        X_scaled=min_max_scaler.transform(df_X)
        X_scaled=pd.DataFrame(X_scaled,
            columns=df_X.columns,index=df_X.index)
        return X_scaled

    @staticmethod
    def balance_data(df_X,df_y):
        """Balance data
        Parameters:
            df_X (DataFrame): Features
            df_y (DataFrame): target.
        Returns:
            df_X_temp (DataFrame): Data balanced.
            df_y_temp (DataFrame): Data balanced.
        """
        ros = RandomOverSampler(random_state=50)
        X_balan_temp, y_balan_temp = ros.fit_resample(df_X, df_y)
        df_X_temp=pd.DataFrame(X_balan_temp,columns=list(df_X))
        df_y_temp=pd.DataFrame(y_balan_temp,columns=list(df_y))
        return df_X_temp, df_y_temp
