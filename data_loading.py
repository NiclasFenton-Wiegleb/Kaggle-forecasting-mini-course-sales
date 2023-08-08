import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import numpy as np

from datasets import Dataset

def load_train_data_label(enc_product, enc_store, enc_country):
    '''Loads and transforms training data set.'''

    df_train = pd.read_csv("train.csv")
    
    df_train["date"] = df_train["date"].str.replace("-","")
    df_train["date"] = pd.to_datetime(df_train["date"], format= "%Y%m%d")    
    df_train["date"] = df_train["date"].dt.to_period("d")
    df_train.set_index(pd.PeriodIndex(df_train["date"], freq="d"), inplace= True)
    df_train.drop("date", axis=1, inplace=True)

    df_train["product"] = enc_product.fit_transform(df_train["product"])
    df_train["store"] = enc_store.fit_transform(df_train["store"])
    df_train["country"] = enc_country.fit_transform(df_train["country"])


    df_train["store"] =  df_train["store"].astype("category")
    df_train["product"] =  df_train["product"].astype("category")
    df_train["country"] =  df_train["country"].astype("category")

    return df_train

def load_train_data_one():
    '''Loads and transforms training data set.'''

    df_train = pd.read_csv("train.csv")
    encoder = OneHotEncoder()
    
    df_train["date"] = df_train["date"].str.replace("-","")
    df_train["date"] = pd.to_datetime(df_train["date"], format= "%Y%m%d")    
    df_train["date"] = df_train["date"].dt.to_period("d")
    df_train.set_index(pd.PeriodIndex(df_train["date"], freq="d"), inplace= True)

    df_train["store"] =  df_train["store"].astype("category")
    df_train["product"] =  df_train["product"].astype("category")
    df_train["country"] =  df_train["country"].astype("category")

    df_category = df_train[["country", "store", "product"]]

    array_category = encoder.fit_transform(df_category).toarray()

    #Assign country onehotencoding
    df_train["Argentina"] = array_category[:,0]
    df_train["Canada"] = array_category[:,1]
    df_train["Estonia"] = array_category[:,2]
    df_train["Japan"] = array_category[:,3]
    df_train["Spain"] = array_category[:,4]

    #Assign store onehotencoding
    df_train["Kagglazon"] = array_category[:,5]
    df_train["Kaggle Learn"] = array_category[:,6]
    df_train["Kaggle Store"] = array_category[:,7]

    #Assign product onehotencoding
    df_train["Using LLMs to Improve Your Coding"] = array_category[:,8]
    df_train['Using LLMs to Train More LLMs'] = array_category[:,9]
    df_train['Using LLMs to Win Friends and Influence People'] = array_category[:,10]
    df_train['Using LLMs to Win More Kaggle Competitions'] = array_category[:,11]
    df_train['Using LLMs to Write Better'] = array_category[:,12]

    #Assign new columns as int
    df_train[["Argentina", "Canada", "Estonia", "Japan", "Spain", "Kagglazon", "Kaggle Learn",
              "Kaggle Store", "Using LLMs to Improve Your Coding", 'Using LLMs to Train More LLMs',
              'Using LLMs to Win Friends and Influence People', 'Using LLMs to Win More Kaggle Competitions',
              'Using LLMs to Write Better']] = df_train[["Argentina", "Canada", "Estonia", "Japan", "Spain", "Kagglazon", "Kaggle Learn",
              "Kaggle Store", "Using LLMs to Improve Your Coding", 'Using LLMs to Train More LLMs',
              'Using LLMs to Win Friends and Influence People', 'Using LLMs to Win More Kaggle Competitions',
              'Using LLMs to Write Better']].astype("int")

    #Remove original columns
    df_train.drop(columns=["store", "product", "country"], axis=1, inplace= True)

    return df_train

def load_test_data(enc_product, enc_store, enc_country):
    '''Loads and transforms the test data set.'''

    #Test dataset
    df_test = pd.read_csv("test.csv")
    
    df_test["date"] = df_test["date"].str.replace("-","")
    df_test["date"] = pd.to_datetime(df_test["date"], format= "%Y%m%d")    
    df_test["date"] = df_test["date"].dt.to_period("d")
    df_test.set_index(pd.PeriodIndex(df_test["date"], freq="d"), inplace= True)
    df_test.drop("date", axis=1, inplace=True)

    df_test["product"] = enc_product.fit_transform(df_test["product"])
    df_test["store"] = enc_store.fit_transform(df_test["store"])
    df_test["country"] = enc_country.fit_transform(df_test["country"])

    df_test["store"] =  df_test["store"].astype("category")
    df_test["product"] =  df_test["product"].astype("category")
    df_test["country"] =  df_test["country"].astype("category")

    return df_test

def save_model(forecaster, file_path):

    model_filename = f"{file_path}.sav"
    joblib.dump(forecaster, model_filename)


if __name__ == "__main__":

    df_train = pd.read_csv("./train.csv")
    encoder = OneHotEncoder()

    df_train["date"] = df_train["date"].str.replace("-","")
    df_train["date"] = pd.to_datetime(df_train["date"], format= "%Y%m%d")
    # df_train["date"] = df_train["date"].dt.to_period("d")
    # df_train.set_index(pd.PeriodIndex(df_train["date"], freq="d"), inplace= True)

    df_train["store"] =  df_train["store"].astype("category")
    df_train["product"] =  df_train["product"].astype("category")
    df_train["country"] =  df_train["country"].astype("category")

    df_category = df_train[["country", "store", "product"]]

    array_category = encoder.fit_transform(df_category).toarray()

    #Assign country onehotencoding
    df_train["Argentina"] = array_category[:,0]
    df_train["Canada"] = array_category[:,1]
    df_train["Estonia"] = array_category[:,2]
    df_train["Japan"] = array_category[:,3]
    df_train["Spain"] = array_category[:,4]

    #Assign store onehotencoding
    df_train["Kagglazon"] = array_category[:,5]
    df_train["Kaggle Learn"] = array_category[:,6]
    df_train["Kaggle Store"] = array_category[:,7]

    #Assign product onehotencoding
    df_train["Using LLMs to Improve Your Coding"] = array_category[:,8]
    df_train['Using LLMs to Train More LLMs'] = array_category[:,9]
    df_train['Using LLMs to Win Friends and Influence People'] = array_category[:,10]
    df_train['Using LLMs to Win More Kaggle Competitions'] = array_category[:,11]
    df_train['Using LLMs to Write Better'] = array_category[:,12]

    #Assign new columns as int

    target = []

    for row in range(len(df_train["id"])):

        num_sold = df_train["num_sold"][row]
        argentina = df_train["Argentina"][row]
        canada = df_train["Canada"][row]
        estonia = df_train["Estonia"][row]
        japan = df_train["Japan"][row]
        spain = df_train["Spain"][row]
        kagglazon = df_train["Kagglazon"][row]
        kaggle_learn = df_train["Kaggle Learn"][row]
        kaggle_store = df_train["Kaggle Store"][row]
        ULtIYC = df_train["Using LLMs to Improve Your Coding"][row]
        ULtTML = df_train["Using LLMs to Train More LLMs"][row]
        ULtWFIP = df_train["Using LLMs to Win Friends and Influence People"][row]
        ULtWMKC = df_train["Using LLMs to Win More Kaggle Competitions"][row]
        ULtWB = df_train["Using LLMs to Write Better"][row]

        target.append(num_sold)
        target.append(argentina)
        target.append(canada)
        target.append(estonia)
        target.append(japan)
        target.append(spain)
        target.append(kagglazon)
        target.append(kaggle_learn)
        target.append(kaggle_store)
        target.append(ULtIYC)
        target.append(ULtTML)
        target.append(ULtWFIP)
        target.append(ULtWMKC)
        target.append(ULtWB)

        # print(df_train[1])
        # print(df[["Argentina", "Canada", "Estonia", "Japan", "Spain", "Kagglazon", "Kaggle Learn",
        #                         "Kaggle Store", "Using LLMs to Improve Your Coding", 'Using LLMs to Train More LLMs',
        #                         'Using LLMs to Win Friends and Influence People', 'Using LLMs to Win More Kaggle Competitions',
        #                         'Using LLMs to Write Better']])

    #Remove original columns
    df_train.drop(columns=["store", "product", "country"], axis=1, inplace= True)

    dates = list(df_train["date"].tolist())

    train_start = []
    train_target = []
    train_feat_static_cat = []
    train_feat_dynamic_real = []
    train_item_id = []

    ticker = 0
    train_id = 1



    for row in range(len(target)):

        if ticker < 136950:
            start = ticker
            end = ticker + 30

            train_target.append(target[start:end])
            train_start.append(dates[start])
            train_feat_static_cat.append([0])
            train_feat_dynamic_real.append(None)
            train_item_id.append(f"T{train_id}")

            ticker += 75
            train_id += 1
        else:
            break

    
    train_dict = {
    "start": train_start,
    "target": train_target,
    "feat_static_cat": train_feat_static_cat,
    "feat_dynamic_real": train_feat_dynamic_real,
    "item_id": train_item_id
    }

    val_start = []
    val_target = []
    val_feat_static_cat = []
    val_feat_dynamic_real = []
    val_item_id = []

    ticker = 30
    val_id = 1



    for row in range(len(target)):

        if ticker < 136950:
            start = ticker
            end = ticker + 45

            val_target.append(target[start:end])
            val_start.append(dates[start])
            val_feat_static_cat.append([0])
            val_feat_dynamic_real.append(None)
            val_item_id.append(f"T{train_id}")

            ticker += 75
            val_id += 1
        else:
            break

    
    val_dict = {
    "start": val_start,
    "target": val_target,
    "feat_static_cat": val_feat_static_cat,
    "feat_dynamic_real": val_feat_dynamic_real,
    "item_id": val_item_id
    }

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)

    print(train_dataset[0])
