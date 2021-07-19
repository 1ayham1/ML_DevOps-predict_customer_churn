''' train ML model to evaluate customer churn'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import shap
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class ChurnLibrarySolution:
    '''import/analyze data and build ML model

    ChurnLibrarySolution class loads .csv data from a data folder, performs exploratory
    data analysis and save results to images/eda folder. It then examins two ML models:
    logistic regression and random forest -after performing grid search to pick up best
    performing parameters

    Training/testing results as well as performance plot are saved to images/results
    '''

    def import_data(self, path):
        '''
        returns dataframe for the csv found at pth

        input:
                pth : a path to the csv (str)
        output:
                df  : pandas dataframe
        '''
        df = pd.read_csv(path)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        return df

    def perform_eda(self, df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        cat_columns = [col for col in df if is_string_dtype(df[col])]
        quant_columns = [col for col in df if is_numeric_dtype(df[col])]

        plt.figure(figsize=(20, 10))
        # ------------------------------------------------------------------
        for q_name in quant_columns:

            sns.histplot(df[q_name], kde=True, stat="density", linewidth=0)
            plt.title(q_name.replace("_", " "), fontsize=18, color="red")
            plt.savefig("./images/eda/NUM_" + q_name + ".jpg")
            plt.clf()
        # ------------------------------------------------------------------
        for c_name in cat_columns:
            category = df[c_name].value_counts()

            labels = category.keys()
            plt.pie(
                x=category,
                autopct="%1.1f%%",
                explode=[0.07] *
                len(labels),
                labels=labels,
                pctdistance=0.5,
                shadow=True,
                startangle=140)
            plt.title(c_name.replace("_", " "), fontsize=18, color="red")
            plt.savefig("./images/eda/CAT_" + c_name + ".jpg")
            plt.clf()

        # ------------------------------------------------------------------
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig("./images/eda/CORR_heatmap.jpg")
        # ------------------------------------------------------------------

    def encoder_helper(self, df, category_lst, response='optional'):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [col of interest]

        output:
                df: pandas dataframe with new columns for
        '''

        for col_name in category_lst:

            encoded_groups = df.groupby(col_name).mean()['Churn']
            encoded_lst = [encoded_groups.loc[val] for val in df[col_name]]

            df[col_name + "_Churn"] = encoded_lst

        return df

    def perform_feature_engineering(self, df, keep_cols):
        '''
        input:
                df: pandas dataframe
                keep_cols: list of string column names to be considered

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        X = pd.DataFrame()
        X[keep_cols] = df[keep_cols]
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return X, X_train, X_test, y_train, y_test

    def classification_report_image(self, y_train,
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
        # scores
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))

        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))

        # -------------------------------------------------------

        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig("./images/results/RF_SummaryReport.jpg")

        # -------------------------------------------------------

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_train, y_train_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_test, y_test_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig("./images/results/LR_SummaryReport.jpg")

    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''

        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 15))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth + "FeatureImortance.jpg")

    def train_models(self, X_data, X_train, X_test, y_train, y_test, trained):
        '''
        train, store model results: images + scores, and store models
        input:
                X_data : features before splitting
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
                trained: flag Ture if the model is already trained.
        output:
                None
        '''
        if not trained:
            # train the model
            rfc = RandomForestClassifier(random_state=42)  # grid search
            lrc = LogisticRegression()

            param_grid = {
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [4, 5, 100],
                'criterion': ['gini', 'entropy']
            }

            cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
            cv_rfc.fit(X_train, y_train)
            rfc_model = cv_rfc.best_estimator_

            lrc.fit(X_train, y_train)
            lr_model = lrc

            # save best model
            joblib.dump(rfc_model, './models/rfc_model.pkl')
            joblib.dump(lr_model, './models/logistic_model.pkl')

        else:
            # load saved model
            rfc_model = joblib.load('./models/rfc_model.pkl')
            lr_model = joblib.load('./models/logistic_model.pkl')

        y_train_preds_rf = rfc_model.predict(X_train)
        y_test_preds_rf = rfc_model.predict(X_test)

        y_train_preds_lr = lr_model.predict(X_train)
        y_test_preds_lr = lr_model.predict(X_test)

        # print scores as images
        self.classification_report_image(y_train, y_test,
                                         y_train_preds_lr, y_train_preds_rf,
                                         y_test_preds_lr, y_test_preds_rf)

        self.feature_importance_plot(rfc_model, X_data, "./images/results/")

        # best model explainer
        plt.figure(figsize=(20, 15))
        explainer = shap.TreeExplainer(rfc_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig("./images/results/BestModel_explainer.jpg")

        plt.clf()
        lrc_plot = plot_roc_curve(lr_model, X_test, y_test)

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc_model,
                                  X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig("./images/results/LRC_ROC.jpg")
