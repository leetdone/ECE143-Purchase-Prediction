import os
import pandas as pd
import argparse

"""
Data Cleaning and Pre-processing Class for Dataset

"""
class DataCleaning:
    def __init__(self, filename, encoding="ISO-8859-1"):
        """
        Constructor for DataCleaning Class
        """
        assert isinstance(filename, str)
        # assert os.path.isfile(filename) == True
        self._df = pd.read_csv(filename, encoding=encoding)
        
    def basic_cleaning(self, col, num_removed=False):
        """
        Removes duplicate and NA/null Customer ID's and duplicates

        :param col: column to remove
        :param num_removed: number of columns to remove
        """
        assert isinstance(self._df, pd.DataFrame)
        
        currentRowEntries = self._df.shape[0]
        # removes rows with NA/null Customer ID's and duplicates
        self._df.dropna(axis = 0, subset = [col], inplace = True)
        self._df.drop_duplicates(inplace = True)
        
        if num_removed: print("Entries removed: {}".format(currentRowEntries - self._df.shape[0]))
            
    def getDataFrame(self):
        """ 
        Getting method for dataframe
        """
        return self._df

    def setDataFrame(self, df):
        """
        Set the dataframe to passed in one

        :param df: valid Pandas dataframe 
        """
        assert isinstance(df, pd.DataFrame)

        self._df = df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="valid e-commerce data")
    args = parser.parse_args()

    dataCleaner = DataCleaning(args.input)
    # print(dataCleaner.getDataFrame().shape)
    
    # print(dataCleaner.getDataFrame().shape)
    
    dataCleaner.basic_cleaning(col='CustomerID', num_removed=True)
    dataCleaner.getDataFrame()['InvoiceDate'] = pd.to_datetime(dataCleaner.getDataFrame()['InvoiceDate'])
    # print(dataCleaner.getDataFrame().shape)

    df_initial = dataCleaner.getDataFrame()
    
    df_cleaned = dataCleaner.getDataFrame().copy(deep = True)
    df_cleaned['QuantityCanceled'] = 0

    entry_to_remove = []

    for index, col in  df_initial.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
        df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                             (df_initial['StockCode']  == col['StockCode']) & 
                             (df_initial['InvoiceDate'] < col['InvoiceDate']) & 
                             (df_initial['Quantity'] > 0)].copy()
        
        df_copy = df_test.copy(deep=True)
        if df_test.shape[0] == 0: 
          entry_to_remove.append(index) 
          continue
        if (df_test.shape[0] > 0): 
            if df_test.shape[0] > 1:
                df_copy = df_test[(df_test['Quantity'] >= -col['Quantity'])]
                if df_copy.shape[0] == 0: continue
                df_copy.sort_index(axis=0 ,ascending=False, inplace = True) 
            if df_copy.shape[0] != 0: 
                df_cleaned.loc[df_copy.index[0], 'QuantityCanceled'] = -col['Quantity']
#                 print('index {}'.format(df_copy.index[0]))
#                 print('item @ df_cleaned[index]: {}'.format(df_cleaned.loc[df_copy.index[0], 'QuantityCanceled']))
        if df_copy.shape[0] != 0: entry_to_remove.append(index)    
        
    print(df_cleaned[df_cleaned['QuantityCanceled'] == 0].shape)
    df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)  
    stockcodes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
    for code in stockcodes:
        df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
    
    print(df_cleaned[df_cleaned['QuantityCanceled'] != 0].shape)
    print(df_cleaned)

    df_cleaned.to_csv('dataprocessed.csv')