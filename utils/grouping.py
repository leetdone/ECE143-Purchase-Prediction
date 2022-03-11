from utils.cluster import encode, preprocess, cluster
import pandas as pd
import copy
from IPython.display import display
from utils.word_extract import word_extracts,list_extract
# from main import product_cluster
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import neighbors, linear_model, svm, tree, ensemble
from utils.word_extract import word_extracts,list_extract
import pandas as pd

def product_cluster(input,word_list):
    '''
    product cluster
    :param input: data to be encoded, dataframe
    :param word_list: attribute of word
    :return: different clusters
    '''
    price_range = [0,1,2,3,5,10]
    prod_encoding = encode(input, word_list, 'product',price_range=price_range)
    prod_encoding = preprocess(prod_encoding,p_type='product')
    prod_cluster = cluster(prod_encoding,
                           n_clusters= 5, 
                           n_init=30, 
                           c_type='product')
    return prod_cluster

def customer_cluster(input,n_cluster,attr_list):
    '''
    product cluster
    :param input: data to be encoded, dataframe
    :param n_cluster: # of clusters
    :param attr_list: attribute we wanna concentrate
    :return: different clusters
    '''
    customer_encoding = encode(input, attr_list, 'customer')
    customer_encoding = preprocess(customer_encoding,p_type='customer')
    custom_cluster = cluster(customer_encoding,
                           n_clusters= n_cluster, 
                           n_init=100, 
                           c_type='customer')
    return customer_encoding,custom_cluster

def customer_data(df_input,description,cluster,cluster_length):
    '''
    preparing the customer data
    :param df_input: data to be encoded, dataframe
    :param description: description of tha data
    :param cluster: different clusters
    :return: different clusters
    '''
    df_data = copy.deepcopy(df_input)
    desc_len = len(cluster)
    prod_cluster = dict()
    for i in range(desc_len):
        prod_cluster[description[i]]=int(cluster[i])
    for i in range(cluster_length):
        new_column = "product_{}".format(i)
        df_data.loc[:,new_column] = 0
    for index, row in df_data.iterrows():
        col = "product_{}".format(int(prod_cluster[row['Description']]))
        price = row['UnitPrice'] * (row['Quantity'] - row['QuantityCanceled'])
        # print(price)
        if price>0: df_data.loc[index,col]=price
    
    temp = df_data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    for i in range(cluster_length):
        column = "product_{}".format(i)
        col_tmp = df_data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[column].sum()
        temp[column] = list(col_tmp[column])
    temp = temp[temp['TotalPrice']>0]
    custom_group = temp.groupby(by=['CustomerID'])['TotalPrice'].agg(['count','mean','sum','min','max'])
    for i in range(cluster_length):
        column = "product_{}".format(i)
        custom_group.loc[:,column] = temp.groupby(by=['CustomerID'])[column].sum()/custom_group['sum']*100
    
    return custom_group

import numpy as np
from sklearn import neighbors, linear_model, svm, tree, ensemble
def model_data(input, attr_list):
    '''
    for predicting the data
    :param input: data data to be changed
    :return: data for train and test
    '''
    dataX = input[attr_list]
    dataY = input['cluster']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataX, dataY, train_size = 0.8)
    return X_train, X_test, Y_train, Y_test

def predict(X_train, X_test, Y_train, Y_test,type):
    '''
    for predicting the data
    :param X_train, X_test, Y_train, Y_test,type: data input
    :return: predicted data
    '''
    if type =="lr":
        model = linear_model.LogisticRegression(max_iter=100)
        param_grid = [
        {'C':np.logspace(0,1,20)}
        ]
        best_model = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)
        best_model.fit(X_train,Y_train)
        return best_model
        # Y_predictions = best_model.predict(X_train)
        # print(100*metrics.accuracy_score(Y_train, Y_predictions))
        # clf = model.fit(X_train,Y_train)