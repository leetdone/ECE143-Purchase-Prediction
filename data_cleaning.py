import os
import pandas as pd

"""
Data Cleaning and Pre-processing Class for Dataset

"""
class DataCleaning:
    def __init__(self, filename, encoding="ISO-8859-1"):
        """
        Constructor for DataCleaning Class
        """
        assert isinstance(filename, str)
        assert os.path.isfile(filename) == True
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
    dataCleaner = DataCleaning('data.csv')
    display(dataCleaner.getDataFrame().shape)
    
    dataCleaner.basic_cleaning(col='CustomerID', num_removed=True)
    display(dataCleaner.getDataFrame().shape)

    df_initial = dataCleaner.getDataFrame()
    df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
    
    df_cleaned = dataCleaner.getDataFrame().copy(deep = True)
    df_cleaned['QuantityCanceled'] = 0

    entry_to_remove = []

    for index, col in  df_initial.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
        df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                             (df_initial['StockCode']  == col['StockCode']) & 
                             (df_initial['InvoiceDate'] < col['InvoiceDate']) & 
                             (df_initial['Quantity']   > 0)].copy()

        if (df_test.shape[0] == 0): 
            entry_to_remove.append(index)
        elif (df_test.shape[0] == 1): 
            index_order = df_test.index[0]
            df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)        
        elif (df_test.shape[0] > 1): 
            df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
            for ind, val in df_test.iterrows():
                if val['Quantity'] >= -col['Quantity']: 
                    df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                    entry_to_remove.append(index) 
                    break    
        
    df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)   
    stockCodes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
    for code in stockCodes:
        df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])

    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
    display(df_cleaned)