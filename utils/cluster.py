''' cluster'''
from concurrent.futures import thread
from matplotlib.pyplot import isinteractive
from pandas import DataFrame
import pandas as pd
from pyparsing import col
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, scale

def encode(input, code_list, e_type,*,price_range=[]):
    '''
    Encode the input data matrix
    :param input: data to be encoded, dataframe
                  For product, it is the dataframe we wanna extract description of products from
                  For customer, it is the dataframe we wanna get customer information with columns 
                  representing attributes
    :param code_list: attribute of data, list
    :param e_type: customer or product encoding, str
    :param price_range: price range of product, list
    :return: encoded data
    '''
    assert isinstance(input, DataFrame)
    assert isinstance(code_list, list)
    assert isinstance(e_type,str)
    assert isinstance(price_range,list)

    if e_type == "product":
        desciption = input['Description'].unique()
        encoding = pd.DataFrame()
        for k, _ in code_list:
            encoding.loc[:,k] = list(map(lambda x: int(k.upper() in x), desciption))
        col_added = []
        if len(price_range)>0:
            for i in range(len(price_range)):
                if i+1 == len(price_range): col_added.append(">{}".format(price_range[i]))
                else: col_added.append("{}-{}".format(price_range[i],price_range[i+1]))
            print(col_added)
            for label in col_added:
                encoding.loc[:,label] = 0
            print("-----There are {} new columns added-----".format(len(col_added)))
            for i,desc in enumerate(desciption):
                price_mean = input[input["Description"]==desc]['UnitPrice'].mean()
                for j in range(len(price_range)):
                    if j+1 == len(price_range): 
                        encoding.loc[i,col_added[j]]=1
                    elif price_mean>price_range[j] and price_mean<=price_range[j+1]:
                        encoding.loc[i,col_added[j]]=1
                        break
                    else: continue

    elif e_type =="customer":
        encoding = input[code_list]
    else:
        raise ValueError('Type not Implemented')
    return encoding

def preprocess(input,p_type):
    '''
    Preprocess before clustering
    :param input: data to preprocessed
    :param p_type: customer or product encoding, str
    :return: preprocessed data
    '''
    assert isinstance(p_type,str)
    pre_data = input.to_numpy()
    if p_type == "customer":
        norm = StandardScaler()
        norm.fit(pre_data)
        pre_data = norm.transform(pre_data)
    return pre_data



def cluster(input, n_clusters, n_init, c_type):
    '''
    Clustering of products or customers using kmeans
    :param input: data encoding, array
    :param n_clusters: number of cluster, int 
    :param n_init: number of initial points for kmeans, int
    :param c_type: customer or product clustering, str
    :return: clustered data
    '''

    assert isinstance(n_clusters, int)
    assert isinstance(n_init,int)
    assert isinstance(c_type,str)
    if c_type == "product":
        silhouette=-100
        while silhouette <= 0.145:
            kmeans_model = KMeans(n_clusters=n_clusters, init ='k-means++', n_init=n_init)
            kmeans_model.fit(input)
            clusters = kmeans_model.predict(input)
            silhouette = silhouette_score(input,clusters)
    elif c_type == "customer":
        kmeans_model = KMeans(n_clusters=n_clusters, init ='k-means++', n_init=n_init)
        kmeans_model.fit(input)
        clusters = kmeans_model.predict(input)
        silhouette = silhouette_score(input,clusters)
    else: 
        raise ValueError('Type not Implemented')
    print("Separate into {} classes with average silhouette score: {}".format(n_clusters, silhouette))
    return clusters