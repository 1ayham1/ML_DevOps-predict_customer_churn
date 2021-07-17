# library doc string
'''
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


class churn_library_solution:

    def __init(self):

        pass
        
        


    def import_data(self):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''	
        
        return pd.read_csv(r"./data/bank_data.csv")


    def perform_eda(self, df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        def plot_figure(data,cols,plt_func):
            '''
            '''
            plt.figure(figsize=(20,10)) 
     
            for q_name in cols:
                
                sns.histplot(df[q_name],kde=True, stat="density", linewidth=0)
                plt.title(q_name.replace("_"," "), fontsize=18, color="red");
                plt.savefig("./images/eda/NUM_"+q_name +".jpg")
                plt.clf()


        cat_columns = [col for col in df if is_string_dtype(df[col])]
        quant_columns = [col for col in df if is_numeric_dtype(df[col])]

     

        plt.figure(figsize=(20,10)) 

        
        for q_name in quant_columns:
            
            sns.histplot(df[q_name],kde=True, stat="density", linewidth=0)
            plt.title(q_name.replace("_"," "), fontsize=18, color="red");
            plt.savefig("./images/eda/NUM_"+q_name +".jpg")
            plt.clf()
        

        for q_name in cat_columns:
            category = df[q_name].value_counts()
            
            labels = category.keys()
            plt.pie(x=category, autopct="%1.1f%%", explode=[0.07]*len(labels), 
                    labels=labels, pctdistance=0.5,shadow=True, startangle=140)
            plt.title(q_name.replace("_"," "), fontsize=18, color="red");
            plt.savefig("./images/eda/CAT_"+q_name +".jpg")
            plt.clf()

        #------------------------------------------------------------------
        #sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        #plt.savefig("./images/eda/corr_heatmap.jpg")


    def encoder_helper(df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        pass


    def perform_feature_engineering(df, response):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

    def classification_report_image(y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        pass


    def feature_importance_plot(model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        pass

    def train_models(X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        pass



if __name__ == '__main__':

    df_obj = churn_library_solution()
    df = df_obj.import_data()
    df_obj.perform_eda(df)

