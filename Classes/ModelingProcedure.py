#For splitting the dataset into train and test.
from sklearn.model_selection import train_test_split
#For keeping the same proportion of target 
# classes while train test splitting.
from sklearn.model_selection import StratifiedKFold
#Linear algebra library
import numpy as np
#DataFrame library
import pandas as pd
#Random generation
import random
#To use K means clustering algorithm
from sklearn.cluster import KMeans
#To automatically calculate the best number of clusters
# based on the elbow method.
from yellowbrick.cluster import KElbowVisualizer
#Auxiliar dictionary operations.
from DictionaryAuxiliar import DictionaryAuxiliar as DA
#General functions for data pre prosessing
from DataStandardizer import DataStandardizer
#Auxliar class for plotting the core machine learning
from PlotterForReport import PlotterForReport
#Interaction between filesystem and python Environment.
# uses pickle.
from OFHandlers import OFHandlers as OFH
#Static class methods for machine learning pre
# pre processing.
from StaticML import StaticML
#Auxiliar static dictionaries to perform machine learning.
from StaticDictML import StaticDictML

#Prevent warnigs to cover the screen.
import warnings
warnings.filterwarnings('ignore')


class MachineLearning:
    """Machine learning main class"""


    def __init__(self,
                df,
                df_cate,
                target,
                string_metric,
                balance_train,
                test_size=0.1):
        """
        Parameters:
            df (DataFrame): Numerical columns
            df_cate (DataFrame): Categorical columns
            target (string): Name of target column
            string_metric (string): Name of metric 
                (see StaticDictML.get_dict_metrics)
            balance_train (boolean): whether to balance
            test_size(float): Size of test data set.
        """
        self.X=df.drop(target,axis=1)
        self.y=df[[target]]
        self.balance_train=balance_train

        #Perform split
        (X_train, X_test,
         y_train, y_test) = train_test_split(self.X, self.y, 
                                            test_size=test_size,
                                            stratify=self.y, 
                                            random_state=42)
        self.X_train, self.X_test=X_train, X_test
        self.y_train, self.y_test= y_train, y_test

        #Get metrics for model evaluation
        self.dict_models=StaticDictML.get_dict_models()
        #Get available machine learning models.
        self.dict_metrics=StaticDictML.get_dict_metrics()
        #Get model evaluation object from user input.
        self.metric=self.dict_metrics.get(string_metric)

        #Scan for categorical variables and perform
        # hot encode on them.
        # save all enconded categorical variables on 
        # self.df_cate
        i=0
        self.df_cate=df_cate
        categorical_not_found=True
        if not df_cate.empty:
            categorical_not_found=False
            print("hot enconding categorical data")
            encoder_objects_dict={}
            for each_col in list(df_cate.columns):
                df_temp=df_cate[[each_col]]
                if(i==0):
                    enconding_holder=DataStandardizer.get_encoded_single_df(df_temp)
                    df_encoded=enconding_holder[0]
                    encoder_objects_dict[each_col]=enconding_holder[1]
                else:
                    enconding_holder_2=DataStandardizer.get_encoded_single_df(df_temp)
                    temp=enconding_holder_2[0]
                    encoder_objects_dict[each_col]=enconding_holder_2[1]
                    print("temp",temp)
                    df_encoded=df_encoded.join(temp,lsuffix=str(i))
                i=i+1
             
            self.encoder_objects_dict=encoder_objects_dict
            self.df_cate=df_encoded
        if categorical_not_found:
            print("Categorical data not detected.")

        self.improved_score_further_op=0
        self.categorical_not_found=categorical_not_found

    def clusterting_metric(self,number_of_clusters=50):
        """Automatic execution of the elbow method"""
        #Min max scale the data
        scaler,XC=StaticML.min_max_df(self.X_train)

        model = KMeans()
        #Perfom elbow method
        visualizer = KElbowVisualizer(model, k=(4,number_of_clusters))
        visualizer.fit(XC)

        #get values of elbow method to plot later
        wcss=pd.DataFrame(visualizer.k_scores_,columns=["wcss"])
        self.wcss=wcss
        #set optimal value of k in clustering.
        self.elbow=visualizer.elbow_value_

    def fit_clustering(self,X):
        """fit data with optimal k founf by elbow method
        Returns:
            clustering (object): fitted cluster.
            cluster_l (DataFrame): Cluster distance space
                transformed data.
        """
        clustering=KMeans(n_clusters=self.elbow,
                        init='k-means++',
                        max_iter=300,
                        n_init=10,
                        random_state=0)
        cluster_labels=["d_"+str(each) for each in range(1,self.elbow+1)]

        cluster_l=pd.DataFrame(clustering.fit_transform(X),
                            index=X.index,
                            columns=cluster_labels)
        return clustering,cluster_l

    def get_recomended_features(self):
        """Use empirical rule of 10 times number
        rows per feature to asses the curse of
        dimensionality.
        Returns:
            reomended_number_features(int): minimum number of features
        """
        shape=self.X_train.shape
        reomended_number_features=int(shape[0]/10)
        return reomended_number_features

    def tran_features(self,X_train,y_train,
                    X_validation,
                    clustering,
                    polynomial):
        """internal pipeline of processing, used mainly on cross validation.
        scale the data. balance the data in agreement with 
        the boolean self.balance_train, also clustering or polynomial.
        Returns:
            X_train (DataFrame): process data on scaling,
                also Raw, clustering, or polynomial.
            X_validation (DataFrame): process data on scaling,
                also Raw, clustering, or polynomial.
            y_train (DataFrame): target class 
        """

        scaler,X_train = StaticML.min_max_df(X_train)
        X_validation = StaticML.transform_min_max(X_validation,scaler)
        #Perform clustering
        if clustering:
            (clustering_obj,
                X_train)=self.fit_clustering(X_train)
            X_validation = StaticML.transform_clustering(X_validation,
                                                        clustering_obj)

        #Perform polynomial
        if polynomial:
            k=self.get_recomended_features()
            best_features=StaticML.select_k_best(X_train,
                                                y_train,
                                                k=k)
            (poly_obj,
                X_train)=StaticML.get_polinomial_df(X_train[best_features])
            X_validation = StaticML.transform_polinomial_df(X_validation[best_features],
                                                            poly_obj)
            best_features_poly=StaticML.select_k_best(X_train,
                                                    y_train,
                                                    k=k)
            X_train=X_train[best_features_poly]
            X_validation=X_validation[best_features_poly]

        #Join numerical data with hot encoded categorical data.
        if not self.categorical_not_found:
            X_train=X_train.join(self.df_cate)
            X_validation=X_validation.join(self.df_cate)

        #Perform balance
        if self.balance_train:
            X_train,y_train=StaticML.balance_data(X_train,y_train)

        return X_train,X_validation,y_train



    def cross_validation(self, model,clustering,polynomial):
        """perform 5 fold cross validation

        Parameters:
            model (object): machine learning algorithm
            clustering (boolean)
            polynomial (boolean)

        Returns:
            validation_scores (list): validations scores
            training_scores (list): training scores
        """
        #StratifiedKFold  keeps the proportions of target classes
        # equal during splitting.
        kf=StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        training_scores=[]
        validation_scores=[]

        for train_index, val_index in kf.split(self.X_train,self.y_train):

            X_train=self.X_train.iloc[train_index]
            y_train=self.y_train.iloc[train_index]

            X_validation=self.X_train.iloc[val_index]
            y_validation=self.y_train.iloc[val_index]

            X_train,X_validation,y_train=self.tran_features(X_train,
                                                            y_train,
                                                            X_validation,
                                                            clustering,
                                                            polynomial)

            model.fit(X_train,y_train)

            #Training scores
            predictions_train = model.predict(X_train)
            training_scores.append(self.metric(y_train, predictions_train))

            #Validation scores
            predictions = model.predict(X_validation)
            validation_scores.append(self.metric(y_validation, predictions))
        return [validation_scores,training_scores]

    def random_search(self, grid, model_string, 
                    clustering, polynomial, max_evals = 2):
        """Performs random search of hyperparameter with 
        the help self.cross_validation
        Parameters:
            grid (dictionary): hyperparameters space
            model_string (string): name of machine learning
                model.
            clustering (boolean) 
            polynomial (boolean)
            max_evals (int): maximun number random selection
                of hyperparameters.
        Returns:
            results (DataFrame): summary of the random search.
        """

        results=pd.DataFrame(columns = ['name_model',
                                        'validation score', 
                                        'params', 
                                        'iteration'],
                            index = list(range(max_evals)))

        for i in range(max_evals):
            hyperparameters={k: random.sample(v, 1)[0] for 
                            k, v in grid.items()}

            model=self.dict_models.get(model_string)

            cv=self.cross_validation(model(**hyperparameters),
                                    clustering,
                                    polynomial)

            eval_results = [model_string,np.mean(cv[0]),
                            hyperparameters,
                            i]

            results.loc[i, :]=eval_results

        results.sort_values('validation score', 
                            ascending = False,
                            inplace = True)

        results.reset_index(inplace = True)

        return results

    def hiper_tunning(self,model_string,clustering,
                    polynomial,
                    max_evals):
        """perform hipertunning"""
        grid=StaticDictML.get_parameter_grid(model_string)
        tunning=self.random_search(grid,
                                model_string,
                                clustering,
                                polynomial,
                                max_evals=max_evals)
        return tunning

    def further_optimizer(self,
                        model_object,
                        best_param_dict_in,
                        clustering,
                        polynomial):
        """search around giving hiperparameters"""
        best_param_dict=best_param_dict_in.copy()
        initial_model=model_object(**best_param_dict_in)
        cv_original=self.cross_validation(model=initial_model,
                                        clustering=clustering,
                                        polynomial=polynomial)

        #Get initial score (before further optimization)
        round_score_original=np.mean(cv_original[0])

        #loop the input hyperparameters
        for k,value in best_param_dict_in.items():
            best_value_step={}
            #check if step value is integer of float,
            # if not, no optimization is donde.
            excluded_keys=["max_iter","random_state","max_iter","nthread","seed"]
            b_excluded_keys=k not in excluded_keys
            b_instance=isinstance(value, int) or isinstance(value, float)
            if b_instance and b_excluded_keys:
                print("walking around ",k)
                #If value is int, then move 5 unist to the right
                # and five unit to the left.
                if isinstance(value, int):
                    localized_parameters=[value+step for step in np.arange(-10,10,1)]
                    if (k=="n_estimators"):
                        localized_parameters=[value+step for step in np.arange(-40,40,4)]

                #If value is float, then test 7 envenltly spaces value between
                # o and tree times value.
                elif isinstance(value, float):
                    localized_parameters=np.linspace(0.0, 6*value, num=17)

                #Loop over localized_parameters
                for each_localized_parameter in localized_parameters:
                    best_param_dict[k]=each_localized_parameter
                    try:
                        print("")
                        #Asses the perfomance with crossvalidation
                        model=model_object(**best_param_dict)
                        cv_scores=self.cross_validation(model=model,
                                                    clustering=clustering,
                                                    polynomial=polynomial)
                    except:
                        #Assign low score to hyperparameters that could not
                        #be processed.
                        cv_scores=[0.2,0.2]
                    round_score=np.mean(cv_scores[0])
                    best_value_step[each_localized_parameter]=round_score
                #Get maximum value of the tested models
                update=DA.key_with_maxval(best_value_step)
                max_round_score=best_value_step.get(update)
                print("candidate update hiperparameters",k,update)
                print("new score:",max_round_score)
                print("old score",round_score_original)
                #if the score is bigger, update the grid of hiperparameters.
                if max_round_score>round_score_original:
                    #Update the original score
                    round_score_original=max_round_score
                    #Update the grid of hiperparameters
                    best_param_dict[k]=update
                else:
                    #keep them the same.grid
                    best_param_dict[k]=best_param_dict_in[k]
        #the overal increase in score by this optimization
        #thi will be sent to the html repor.
        increase_metric=abs(round_score_original-np.mean(cv_original[0]))*100
        return best_param_dict,model_object,increase_metric

    def produce_final_test(self,super_model,super_parameters,
                        clustering,
                        polynomial,
                        mode="tunned"):
        """Fit model on test data
        Returns:
            final_training_score (float): traning score
            expected_score (float): test score
        """

        X_train, X_validation=self.X_train, self.X_test
        y_train, y_validation=self.y_train, self.y_test
        #Transform data using best parameters of modeling
        X_train,X_validation,y_train=self.tran_features(X_train,
                                                    y_train,
                                                    X_validation,
                                                    clustering,
                                                    polynomial)
        #Fit model
        model=super_model
        if(mode=="tunned"):
            model=model(**super_parameters)
        else:
            model=model()
        model.fit(X_train,y_train)

        #Training scores
        predictions_train = model.predict(X_train)
        final_training_score=self.metric(y_train, predictions_train)

        #Validation scores
        predictions = model.predict(X_validation)
        expected_score=self.metric(y_validation, predictions)
        return final_training_score,expected_score


    def get_base_model(self,clustering,polynomial):
        """Run cross validation on with defualt parameters"""
        base_model={}
        random_state=42
        for key,each_model in self.dict_models.items():
            #Try to set random_state state parameters
            # to ensure reproductability.
            try:
                if clustering:
                    base_model[key]=self.cross_validation(
                                    each_model(random_state=random_state),
                                    True,
                                    False)
                elif polynomial:
                    base_model[key]=self.cross_validation(
                                    each_model(random_state=random_state),
                                    False,
                                    True)
                else:
                    base_model[key]=self.cross_validation(
                                    each_model(random_state=random_state),
                                    False,False)
            except:
                if clustering:
                    base_model[key]=self.cross_validation(
                                    each_model(),
                                    True,
                                    False)
                elif polynomial:
                    base_model[key]=self.cross_validation(
                                    each_model(),
                                    False,
                                    True)
                else:
                    base_model[key]=self.cross_validation(
                                    each_model(),
                                    False,False)

        return base_model

    def configurations_and_plot(self,number_of_clusters,
                            clustering,polynomial,working_directory):
        """test with default parameters
         on three available modes:
           Raw data
        ...Dimension reduced by clustering.
        ...polynomial features.
         Note that the available modes are visited in agreement to 
         the boolean inputs clustering and polynomial.
         On the best mode, it selects the best model using
         the highest cross-validation error, and model with
         with smallest standard deviation for further optimization
         (random search of set of hyperparameters).
         It also saves the corresponding plots on the ./name/img 
         folder.
         Note 2: clustering and polynomial booleans get updated 
         and are not longer the ones set by the user.
        """

        score_mean_dict={}
        boolean_selection_dict={}
        auxiliar_dict={}

        #Test on raw data
        base_models_scores_nc=self.get_base_model(clustering=False,polynomial=False)
        max_mean_cv_validation=DA.get_max_mean(base_models_scores_nc)
        msg="max_mean_cv_validation for"
        print(msg+" no-clustering no-polynomial",max_mean_cv_validation)
        score_mean_dict[0]=max_mean_cv_validation
        boolean_selection_dict[False,False]=max_mean_cv_validation
        auxiliar_dict[0]=base_models_scores_nc

        #Test using k means distance space
        if(clustering==True):
            self.clusterting_metric(number_of_clusters=number_of_clusters)
            base_models_scores_c=self.get_base_model(clustering=clustering,
                                                    polynomial=False)
            PlotterForReport.save_clustering_plot(self.wcss,
                            self.elbow,
                            path_temp_images=working_directory+"/img")
            max_mean_cv_validation=DA.get_max_mean(base_models_scores_c)
            print(msg+" clustering",max_mean_cv_validation)
            score_mean_dict[1]=max_mean_cv_validation
            boolean_selection_dict[clustering,False]=max_mean_cv_validation
            auxiliar_dict[1]=base_models_scores_c

        #Test using polynomial
        if(polynomial==True):
            base_models_scores_poly=self.get_base_model(clustering=False,
                                                    polynomial=polynomial)
            #print("base_models_scores_poly",base_models_scores_poly)
            max_mean_cv_validation=DA.get_max_mean(base_models_scores_poly)
            print(msg+" for polynomial",max_mean_cv_validation)

            score_mean_dict[2]=max_mean_cv_validation

            boolean_selection_dict[False,polynomial]=max_mean_cv_validation

            auxiliar_dict[2]=base_models_scores_poly

        print("score_mean_dict",score_mean_dict)

        # get the maximum configuration results based on the
        # maximun cv validation against each other here, the key 
        # is another dictionary "model":[list_validation,list_training]
        best_model_config= auxiliar_dict.get(DA.key_with_maxval(score_mean_dict))
        clustering,polynomial= DA.key_with_maxval(boolean_selection_dict)

        #save best line report
        PlotterForReport.save_base_line_plot(best_model_config,
                            path_temp_images=working_directory+"/img")
        str_model_best_score=DA.key_with_maxval({k:np.mean(v[0]) for
                                    k,v in best_model_config.items()})
        str_model_min_std=DA.key_with_minval({k:np.std(v[0]) for
                                    k,v in best_model_config.items()})
        return str_model_best_score,str_model_min_std,clustering,polynomial,score_mean_dict
    

    def further_optimize_pipeline(self,cvb_best_score_tunning,
                                cvb_best_std_tunning,
                                futher_op_boo,
                                clustering,
                                polynomial):
        """run optimization pipeline around best tunned
        parameters, here the final model is selected 
            cvb_best_score_tunning (DataFrame): parameters 
                and best machine learning model on base line
                based on cross valídation.
            cvb_best_std_tunning (DataFrame): parameters 
                and best machine learning model on base line
                based on minimum standard deviation.
            futher_op_boo (boolean): whether to perform a 
                step walk arounf hiperparameters.
            clustering (boolean)
            polynomial (boolean)
        """
        # Best score on cv
        model_1_score=cvb_best_score_tunning['validation score'][0]
        # Best score on minimum std
        model_2_score=cvb_best_std_tunning['validation score'][0]
        print("before walking around hyperparameters")

        #Print name of best models and prior walking optimization
        print(cvb_best_score_tunning.name_model[0],model_1_score)
        print(cvb_best_std_tunning.name_model[0],model_2_score)

        #Perform a pre selection of best model
        if model_2_score>model_1_score:
            further_tunning_params=cvb_best_std_tunning.params[0]
            string_model=cvb_best_std_tunning.name_model[0]
            pre_score=model_2_score
        else:
            further_tunning_params=cvb_best_score_tunning.params[0]
            string_model=cvb_best_score_tunning.name_model[0]
            pre_score=model_1_score
        
        model_object=self.dict_models.get(string_model)

        #Check whether conditions for walking optimization 
        # are met. Note that even if futher_op_boo=True
        # walking optimization also requires that the pre
        # selected score is smaller than 0.81.
        # If this check is true, the computational resourses 
        # as well as computational time will go up greatly.
        if(futher_op_boo and pre_score<0.81):

            # Walk optimize tunned best model on cv score
            print("first futher op")
            model1=self.dict_models.get(cvb_best_score_tunning.name_model[0])
            (best_param_dict_1,
                best_object_1,
                improved_score_1)=self.further_optimizer(model_object=model1,
                    best_param_dict_in=cvb_best_score_tunning.params[0],
                    clustering=clustering,polynomial=polynomial)

            # Walk optimize tunned best model on minimum std
            print("second futher op")
            model2=self.dict_models.get(cvb_best_std_tunning.name_model[0])
            (best_param_dict_2,
                best_object_2,
                improved_score_2)=self.further_optimizer(model_object=model2,
                    best_param_dict_in=cvb_best_std_tunning.params[0],
                    clustering=clustering,polynomial=polynomial)

            #Test all available models on the test set.
            # Posible outcomes:
            # 1. best tunned model based on cv validation
            # 2. best tunned model based on minimun std
            # 3. first model (with default parameters)
            # 4. second model (with default parameters)
            print("calculating candidates on test set:")
            print("1. best tunned model based on cv validation")
            print("2. best tunned model based on minimun std")
            print("3. first model (with default parameters)")
            print("4. second model (with default parameters)")

            (final_training_score_1,
                expected_score_1)=self.produce_final_test(best_object_1,
                                                    best_param_dict_1,
                                                    clustering,polynomial)
            (final_training_score_2,
                expected_score_2)=self.produce_final_test(best_object_2,
                                                    best_param_dict_2,
                                                    clustering,polynomial)

            (final_training_score_3,
                expected_score_3)=self.produce_final_test(best_object_1,
                                                    None,
                                                    clustering,polynomial,mode="default")

            (final_training_score_4,
                expected_score_4)=self.produce_final_test(best_object_2,
                                                    None,
                                                    clustering,polynomial,mode="default")

            print("expected_score_1,expected_score_2",expected_score_1,expected_score_2)
            print("expected_score_3,expected_score_4",expected_score_3,expected_score_4)

            #dictionary of scores
            results_on_test_dict={1:expected_score_1,
                    2:expected_score_2,
                    3:expected_score_3,
                    4:expected_score_4,
                    }
            print("results_on_test_dict",results_on_test_dict)

            #Create data structures for the candidates
            list_m_1=[best_param_dict_1,
                    final_training_score_1,
                    expected_score_1,cvb_best_score_tunning.name_model[0],
                    improved_score_1]

            list_m_2=[best_param_dict_2,
                    final_training_score_2,
                    expected_score_2,cvb_best_std_tunning.name_model[0],
                    improved_score_2]

            list_m_3=[None,
                    final_training_score_3,
                    expected_score_3,cvb_best_score_tunning.name_model[0],
                    None]

            list_m_4=[None,
                    final_training_score_4,
                    expected_score_4,cvb_best_std_tunning.name_model[0],
                    None]

            #Dictionary of data structures of the four candidates
            results_on_test_list={1:list_m_1,
                                2:list_m_2,
                                3:list_m_3,
                                4:list_m_4,
                                }

            #Get the key of model with best score.
            selected_key=DA.key_with_maxval(results_on_test_dict)
            print("selected_key",selected_key)

            l=results_on_test_list.get(selected_key)
            print("l",l)
            self.super_parameters=l[0]
            self.final_training_score=l[1]
            self.expected_score=l[2]
            self.string_model=l[3]
            self.improved_score_further_op=l[4]
        else:
            final_training_score,expected_score=self.produce_final_test(model_object,
                                                further_tunning_params,
                                                clustering,polynomial)
            self.super_parameters=further_tunning_params
            self.final_training_score=final_training_score
            self.expected_score=expected_score
            self.string_model=string_model
            self.improved_score_further_op=None




    def tran_predictions(self,X,y,clustering,
                        polynomial,working_directory,name):
        """Pipeline to transform features into
        the scale, clustering or polynomial space 
        used on full data set. Train and Test. It also
        saves the pertinent pickle file in disk for
        later production deployment.

        Return:
            X (DataFrame): Transformed features
            y (DataFrame): target.
        """
        scaler,X = StaticML.min_max_df(X)
        print("saving scaler at", working_directory+name+"_scaler.file")
        OFH.save_object(working_directory+name+"_scaler.file",scaler)

        #fix bug on categorical data
        if not self.categorical_not_found:
            print("saving Categorical encoders", 
                    working_directory+name+"_scaler.file")
            OFH.save_object(working_directory+
                            name+"_catencoders.file",
                            self.encoder_objects_dict)
            

        if clustering:
            clustering_obj,X=self.fit_clustering(X)
            OFH.save_object(working_directory+name+
                            "_clustering_obj.file",clustering_obj)

        if polynomial:
            best_features=StaticML.select_k_best(X,y,
                                        k=self.get_recomended_features())
            poly_obj,X=StaticML.get_polinomial_df(X[best_features])
            best_features_poly=StaticML.select_k_best(X,y,
                                        k=self.get_recomended_features())
            X=X[best_features_poly]

            OFH.save_object(working_directory+name+
                                    "_best_features.file",best_features)
            OFH.save_object(working_directory+name+
                                    "_best_features_poly.file",best_features_poly)
            OFH.save_object(working_directory+name+
                                    "_poly_obj.file",poly_obj)

        #Merge with categorical features if
        # they exits
        if not self.categorical_not_found:
            X=X.join(self.df_cate)

        if self.balance_train:
            X,y=StaticML.balance_data(X,y)

        return X,y

    def build_final_clf(self,clustering,polynomial,
                        working_directory,name):
        """Build classfier over the entire
        dataset (test and train) to prepare it
        to be exported to production.
        """

        X_train, X_validation=self.X_train, self.X_test
        y_train, y_validation=self.y_train, self.y_test

        super_X=pd.concat([X_train, X_validation])
        super_y=pd.concat([y_train, y_validation])

        #Transform data into the apropiate space
        super_X,super_y=self.tran_predictions(super_X,
                                            super_y,
                                            clustering,
                                            polynomial,
                                            working_directory,
                                            name)


        model_object=self.dict_models.get(self.string_model)
        parameters=self.super_parameters
        #If parameters=None, default configuration
        try:
            model=model_object(**parameters)
        except:
            model=model_object()

        model.fit(super_X,super_y)

        self.trained_model=model

        #Training scores
        predictions_train = model.predict(super_X)
        trained_scored=self.metric(super_y, predictions_train)

        #Plot the roc curve
        fpr,tpr,auc_score=StaticDictML.get_roc_curve(super_y,
                                                    predictions_train)

        PlotterForReport.save_roc_plot(fpr,tpr,auc_score,
                            super_y,
                            path_temp_images=working_directory+"/img")

        self.trained_scored=trained_scored


