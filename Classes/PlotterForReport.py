
# Dataframe library
import pandas as pd
# Plotting library
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Linear algebra
import numpy as np
#Interaction between filesystem and python Environment.
# uses pickle.
from OFHandlers import OFHandlers as OFH
#Roc plot
from sklearn.metrics import roc_curve

#Set plot style
mpl.style.use(['ggplot']) 
#Set color palette for plots 
colors_list=sns.color_palette("Blues", 10)


class PlotterForReport:
    """Plots for AML_MS report"""

    def __init__(self):
        pass

    @staticmethod
    def save_clustering_plot(wcss,elbow,
                            path_temp_images="./temp_images"):
        """Plot the elbow method
        for selecting the optimal k in
        k means clustering.

        Parameters:
            wcss (list) : Within cluster sums of squares values
            elbow (int) : Optimal k position.
            path_temp_images (string) : folder location to save
                plot.
        """
        x=np.arange(1,len(wcss),0.5)
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,7))
        wcss.plot(kind='line', ax=ax,label="Discrete",style='.-')

        # Annotate arrow
        ax.annotate('Selected K '+str(elbow),
                     xy=(elbow,wcss.iloc[elbow]),
                     xytext=(elbow,wcss.iloc[elbow]*1.03),
                     xycoords='data',
                     arrowprops=dict(arrowstyle='->', 
                                    connectionstyle='arc3',
                                    color='blue', lw=2)
                    )

        ax.set_title ('The elbow method')
        ax.set_ylabel('WCSS')
        ax.set_xlabel('k Cluster')
        ax.legend(loc='best')
        ax.set_xlim(1,len(wcss))
        fig.savefig(path_temp_images+"/clustering_report.png",
                    bbox_inches='tight')


    @staticmethod
    def save_base_line_plot(base_models_scores,
                            path_temp_images="./temp_images"):
        """Plot the a box plot and heatmap
        of the machine learning models with default 
        hyperparameters.
        Paramaters:
            base_models_scores (dictionary)
            path_temp_images (string) : folder location to save
            plot.
        """

        cv_mean_base_model={k:[np.mean(v[0]),
                            np.mean(v[1])] for k,v in 
                            base_models_scores.items()}

        base_models=pd.DataFrame(cv_mean_base_model,
                                index=["CV Validation","CV Training"])
        Validations_scores={k:v[0] for k,v in 
                            base_models_scores.items() }
        Validations_scores=pd.DataFrame(Validations_scores)

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
        msg1='Distribution and Results of '
        msg2='Cross Validation on default hyperparameters'
        fig.suptitle(msg1+msg2, fontsize=16)
        ax1, ax2= ax.flatten()

        Validations_scores.plot(kind='box',ax=ax1)
        ax1.set_ylabel("CV Validation Error")
        ax1.set_xlabel("Tested Models")

        ax2 = sns.heatmap(base_models.T[["CV Validation","CV Training"]],
                          annot=True, fmt="f", 
                          linewidths=.01,cmap=colors_list)
        ax2.set_xlabel("CV Scores")
        ax2.set_ylabel("Tested Models")
        fig.savefig(path_temp_images+"/base_line.png",
                    bbox_inches='tight')


    @staticmethod
    def save_roc_plot(fpr,tpr,auc_score,
                    y_train,path_temp_images="./temp_images"):
        """Plot roc curve and class distribution and saves to folder.
        Paramaters:
            fpr (list) : false positives rate
            tpr (list) : true porsitives rate
            y_train (DataFrame) : Target class
            path_temp_images (string) : folder location
        """

        fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
        ax1,ax2=ax.flatten()

        # roc curve
        ax1.plot(fpr,tpr,color=colors_list[9],
                label="Auc score "+str(auc_score))
        ax1.plot([0, 1], [0, 1], color=colors_list[3],
                linestyle="--",label="unskilled classifier")

        ax1.set_title("ROC curve Selected model", y=1.05) 
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc='lower right')

        y_train["Classes"]=y_train[y_train.columns[0]]
        v=y_train.groupby(y_train.columns[0]).count()

        #Pie chart
        v["Classes"].plot(kind="pie",
                        figsize=(15, 6),
                        autopct="%1.1f%%", 
                        startangle=90,    
                        shadow=True,       
                        labels=None,         
                        pctdistance=1.12,
                        colors=[colors_list[0],colors_list[9]]
                        )
        ax2.set_title("Class Distribution (training-set)", y=1.05) 
         
        # add legend
        ax2.legend(labels=v.index, loc='upper left')
        fig.savefig(path_temp_images+"/roc_plot.png", 
                    bbox_inches='tight')



