from utils.word_extract import word_extracts,list_extract
# from main import product_cluster
from utils.grouping import product_cluster,customer_data,model_data,predict,customer_cluster
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import neighbors, linear_model, svm, tree, ensemble
import argparse

if __name__ == '__main__':
    # product grouping
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_init",default="./data.csv",help="valid e-commerce data")
    parser.add_argument("--input_cleaned",default="./dataprocessed.csv",help="cleaned e-commerce data")
    args = parser.parse_args()
    txt_init_path = args.input_init
    txt_cleaned_path = args.input_cleaned
    df_initial = pd.read_csv(txt_init_path, encoding = "ISO-8859-1", dtype={'InvoiceID':str,'CustomerID':str})
    df_cleaned = pd.read_csv(txt_cleaned_path, encoding = "ISO-8859-1", dtype={'InvoiceID':str,'CustomerID':str})
    prod_desc = pd.DataFrame(df_initial['Description'].unique())
    prod_desc = prod_desc.rename(columns={0:'Description'})
    key_count = word_extracts(prod_desc,"Description")
    word_list = list_extract(key_count)
    prod_cluster = product_cluster(df_cleaned,word_list)

    #customer grouping
    description = df_cleaned['Description'].unique()
    print(len(description))
    customer_predata = customer_data(df_cleaned,description,prod_cluster,5)
    customer_predata.reset_index(inplace=True, drop=False)

    attr_list = customer_predata.columns.values.tolist()
    customer_encoding, custom_cluster = customer_cluster(customer_predata,7,attr_list)
    customer_predata.loc[:,"cluster"] = custom_cluster

    #model prediction
    attr_list = ['mean', 'product_0', 'product_1', 'product_2', 'product_3', 'product_4']
    X_train, X_test, Y_train, Y_test=model_data(customer_predata,attr_list)
    best_model = predict(X_train, X_test, Y_train, Y_test,"lr")
    print("{}%".format(best_model.best_score_*100))



