# Dataframe library
import pandas as pd
# Imputer, for fill in NaN
from sklearn.preprocessing import Imputer
# Library for enconding categorical variables
from sklearn.preprocessing import OneHotEncoder
# Lineas algebra library
import numpy as np


class DataStandardizer:
    """General class of pre processing functions."""


    def __init__(self):
        pass

    @staticmethod
    def fix_columns_and_decimal(df):
        """Ajust the columns names to a secuential
        of string.Fix the decimal from commas to dots.

        Parameters:
            df (DataFrame): DataFrame of dataset.

        Returns:
            df_temp (DataFrame): Ajusted columns.
            mapper_cols (dictionary): key prior column's name
                                    value ajusted name.
        """
        cols=list(df.columns)
        mapper_cols={str(index):each_col for index,each_col in enumerate(cols)}
        df_ajusted=pd.DataFrame(df.values, 
                                columns=list(mapper_cols.keys()),
                                dtype=df.values.dtype)
        df_temp=df_ajusted.copy()
        for every_col in list(df_ajusted.columns):
            try:
                df_temp[every_col] = [x.replace(',', '.') 
                                    for x in df_ajusted[every_col]]
                df_temp[every_col] = df_temp[every_col].astype(float)
            except:
                try:
                    df_temp[every_col] = df_temp[every_col].astype(float)
                except:
                    pass
        return df_temp,mapper_cols

    @staticmethod
    def get_numerics_cols(df):
        """Returns DataFrame of only numeric columns"""
        numerics = ['int16', 'int32', 'int64', 
                    'float16', 'float32', 'float64']
        df_numeric = df.select_dtypes(include=numerics)
        return df_numeric

    @staticmethod
    def replace_nan(df):
        """Returns DataFrame with no NaN values"""
        imputer = Imputer(missing_values= 'NaN', 
                        strategy = 'mean',axis=0)
        imputer_values=imputer.fit_transform(df)
        df_nonan=pd.DataFrame(imputer_values,
                            columns=df.columns,index=df.index)
        return df_nonan

    @staticmethod
    def get_binary_target(df_ajusted,target_col):
        """Returns ajusted binary target: 0,1"""
        df=df_ajusted[[target_col]].copy()
        labels=list(df[target_col].unique())
        labels.sort()
        df.replace(labels[0], 0,inplace=True)
        df.replace(labels[1], 1,inplace=True)
        mapped_label={labels[0]:0,labels[1]:1}
        return df,mapped_label

    @staticmethod
    def get_null_coll(df):
        """Returns list of columns with missing values"""
        null_cols_list=df.isna().any()
        return null_cols_list

    @staticmethod
    def get_encoded_single_df(data_frame_column):
        """one hot encoded a single column dataframe.
        Parameters:
            data_frame_column (DataFrame): single column.

        Returns:
            df_encoded_data (DataFrame):hot encoded.
            encoder_object (object): encoder object.
        """
        df_initial=data_frame_column.copy()
        encoder_object=OneHotEncoder(sparse=False,
                                    categories='auto',
                                    drop='first')
        data_frame_column=np.array(data_frame_column)
        data_frame_column=data_frame_column.reshape(-1, 1)
        #print(data_frame_column.shape)
        encoded_data = encoder_object.fit_transform(data_frame_column)
        cat=encoder_object.categories_
        print("----",cat[0])
        cat=cat[0][1:]
        print("----",cat)
        df_encoded_data=pd.DataFrame(encoded_data,
                                    columns=cat,
                                    index=df_initial.index)
        return[df_encoded_data,encoder_object]

    @staticmethod
    def transform_encoded_single_df(data_frame_column,encoder_object):
        """transform data_frame_column into encoded version
            using an encoder_object
        """
        df_initial=data_frame_column.copy()
        data_frame_column=np.array(data_frame_column)
        data_frame_column=data_frame_column.reshape(-1, 1)
        encoded_data = encoder_object.transform(data_frame_column)
        df_encoded_data=pd.DataFrame(encoded_data,
                                    columns=encoder_object.categories_,
                                    index=df_initial.index)
        return df_encoded_data

    def pipe_line_stand(self,df,mode="target"):
        """ Pre processing pipeline

        Parameters:
            df (DataFrame): Full dataset
            mode (string): Indicates if the target column
                is included.

        Returns:
            df (DataFrame): Processed DataFrame of numeric columns
            df_cate (DataFrame): Categorical columns data.
            target_1 (DataFrame): Target single column.
        """
        
        # Remove register with null targets
        if mode=="target":
            list_col=list(df.columns)
            df.dropna(subset=[list_col[-1]],inplace=True)

        df,maped_cols_1=self.fix_columns_and_decimal(df)

        if mode=="target":
            target_1=str(len(maped_cols_1)-1)
            Y_1,maped_labels_1=self.get_binary_target(df,target_1)
            df.drop(target_1,axis=1,inplace=True)
        original_cols=list(df.columns)
        df_hols=df.copy()
        # use imputer
        df=self.replace_nan(df)
        # get numerical data
        df=self.get_numerics_cols(df)

        categorical_columns=[col for col in 
                            original_cols if col not in list(df.columns)]

        df_cate=df_hols[categorical_columns]

        #Remove register with null values on
        # on categorical data.
        if(len(categorical_columns)>0):
            df_cate.dropna(inplace=True)
            index_based_cate=df_cate.index
            df=df.iloc[index_based_cate]
        if mode=="target":
            df[target_1]=Y_1
            return df,df_cate,target_1
        else:
            return df,df_cate





