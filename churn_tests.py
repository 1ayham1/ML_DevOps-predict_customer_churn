#import os
import time
import logging
import unittest


from pandas.api.types import is_string_dtype
from functools import wraps

from churn_library import ChurnLibrarySolution


keep_cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    #filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def get_time(function):
    '''
    wrapper to return execution time of a function
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        run_fun = function(*args, **kwargs)
        t_end = time.time() - t_start
        logging.info(f'{function.__name__} ran in {t_end:0.3f} sec')
        logging.info(f'{"-"*60}')
        
        return run_fun
    return wrapper
    
class TestingAndLogging(unittest.TestCase):

    def setUp(self):
        self.churn_obj = ChurnLibrarySolution()
        #needs refactoring so that churn class asserts read file correctness
        self.df = self.churn_obj.import_data("./data/bank_data.csv")
        self.category_lst = [col for col in self.df if is_string_dtype(self.df[col])]
  
    @get_time
    def test_import(self):
        '''
        test data import
        '''
        
        try:
            df = self.churn_obj.import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS")
     
        except FileNotFoundError as err:
            logging.error("Testing import_data: file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            logging.info(f"churn data has {df.shape[0]} rows and {df.shape[1]} columns ")

        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    @get_time
    def test_eda(self):
        '''test perform eda function'''
  
        try:
            self.churn_obj.perform_eda(self.df)
            logging.info(f"exploratory figures has been generated. check [./images/eda/] ")
        
        except Exception as err:
            logging.error(f"someting is wrong with eda()")
            raise err

    @get_time
    def test_encoder_helper(self):
        '''
        test encoder helper
        '''
        try:
            df = self.churn_obj.encoder_helper(self.df, self.category_lst)
            logging.info(f"encoder helper ran successfully. ")
        
        except Exception as err:
            logging.error(f"someting is wrong with encoder_helper()")
            raise err

    @get_time
    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''
        df = self.churn_obj.encoder_helper(self.df, self.category_lst)

        try:
            X_data, X_train, X_test, y_train, y_test = self.churn_obj.perform_feature_engineering(df,keep_cols)
            logging.info(f"Train/Test and feature engineering ran successfully!")
            logging.info(f"X_train is of shape: {X_train.shape}")
            logging.info(f"Y_train is of shape: {y_train.shape}")
            logging.info(f"X_test is of shape: {X_test.shape}")
            logging.info(f"Y_test is of shape: {y_test.shape}")
        
        except Exception as err:
            logging.error(f"someting is wrong with perform_feature_engineering()")
            raise err
        
    @get_time
    def test_train_models(self):
        '''
        test train_models
        '''
        df = self.churn_obj.encoder_helper(self.df, self.category_lst)
        X_data, X_train, X_test, y_train, y_test = self.churn_obj.perform_feature_engineering(df,keep_cols)

        try:
            self.churn_obj.train_models(X_data,X_train, X_test, y_train, y_test,trained=True)
            logging.info(f"Train/Test and feature engineering ran successfully!")
                  
        except Exception as err:
            logging.error(f"someting is wrong with train_models()")
            raise err


if __name__ == '__main__':

    unittest.main()
    
    #churn_obj = ChurnLibrarySolution()
    #df = churn_obj.import_data("./data/bank_data.csv")
    #category_lst = [col for col in df if is_string_dtype(df[col])]
    #df = churn_obj.encoder_helper(df, category_lst)
    #X_data, X_train, X_test, y_train, y_test = churn_obj.perform_feature_engineering(df,keep_cols)
    #churn_obj.train_models(X_data,X_train, X_test, y_train, y_test,trained=True)


