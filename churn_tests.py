import os


from pandas.api.types import is_string_dtype
from functools import wraps

from churn_library import ChurnLibrarySolution


class TestingAndLogging:

    def __init__(self):
        pass

    def get_log_info(test_func):

        import logging

        logging.basicConfig(
            filename='./logs/churn_library.log',
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')
		
		@wraps(test_func)
        return None

    def test_import(self, import_data):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        try:
            df = import_data("./data/bank_data.csv")
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_eda(perform_eda):
        '''
        test perform eda function
        '''

    def test_encoder_helper(encoder_helper):
        '''
        test encoder helper
        '''

    def test_perform_feature_engineering(perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''

    def test_train_models(train_models):
        '''
        test train_models
        '''


if __name__ == '__main__':

    df_obj = ChurnLibrarySolution()
    test_obj = TestingAndLogging()

    test_obj.test_import(df_obj.import_data)

    #df = df_obj.import_data("./data/bank_data.csv")
    # print(df.head())

    # df_obj.perform_eda(df)
    #category_lst = [col for col in df if is_string_dtype(df[col])]

    #df = df_obj.encoder_helper(df, category_lst)

    #X_train, X_test, y_train, y_test = df_obj.perform_feature_engineering(df)
    # print(X_train.head())
