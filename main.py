from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.regression import *
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def uploadfile(file):
    file_extension = file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        df = pd.read_excel(file)
    elif file_extension == "json":
        df = pd.read_json(file)
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
        return None
    df.head()
    return df
# function for EDA and data summary
def EDA(data):
    st.subheader("Exploratory Data Analysis")

    #statistics
    st.write("Summary Statistics:")
    st.write(data.describe())

    #visualization
    st.write("Data Visualization:")
    #check the data types of columns
    numericCols = data.select_dtypes(include=np.number).columns
    categoricalCols = data.select_dtypes(include='object').columns
    #pairplot for numeric columns
    st.write("pairplot for numeric columns:")
    sns.pairplot(data[numericCols])
    st.pyplot()
    #count plots for categorical columns
    st.write("count plot for categorical columns (if any): ")
    for col in categoricalCols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data[col])
        plt.title(f"{col} Count Plot")
        plt.xlabel(col)
        plt.ylabel("Count")
        #show on web app
        st.pyplot()
#function to split data   
def dataSplit(data):
    data_sample = data.sample(frac=0.95, random_state=786)
    data_unseen = data.drop(data_sample.index)
    return data_sample,data_unseen
#function set up data  
def dataSetup(data, targetVar):
    unique_values = data[targetVar].unique()
    #if there is binery uniqe values
    if len(unique_values) <= 2:  
        problemType = 'classification'
        st.write('classification')
        #find categorical data
        columns_with_0_1 = []
        for column in data.columns:
            if all(value in [0, 1] for value in data[column]):
                columns_with_0_1.append(column)
        columns_with_0_1.remove(targetVar)
        st.write(columns_with_0_1)
        return problemType, setup(data=data, target=targetVar, categorical_features=columns_with_0_1, session_id=123) 
    # If the target variable is regression
    elif data[targetVar].dtype in ['int64', 'float64']:
        problemType = 'regression'
        st.write('regression')
        return problemType, setup(data=data, target=targetVar, session_id=123)

    else:
        st.write('Unable to determine the type of problem. Further inspection may be needed.')
        return None, None
    
#show predections
def test_predict_model (model, data):
    predictions = predict_model(model,data=data)
    st.write(predictions.head())

# MAIN

#app title
st.title("Machine Learning Model")
#add a header
st.header("Input Parameters")
#prompt the user to upload the dataset
file_path = st.file_uploader("input file name with extention")
#target variable
target_column_name = st.text_input("Enter the name of the target column:", value='target Column')
#colomns to delete
delete_col = st.text_input("Enter column name you want to delete (optional) : ", value='delete columns')
submit = st.button("Submit")

# process input when its Submited
if submit:
    data = uploadfile(file_path)
    if delete_col and delete_col in data.columns:
        data.drop(columns=delete_col, inplace=True)
    st.write(data)
    st.write(data.shape)
    EDA(data)
    data_sample, data_test_set = dataSplit(data)
    st.write('sample : ')
    st.write(data_sample)
    st.write('sample shape : ')
    st.write(data_sample.shape)
    st.write('test sample : ')
    st.write(data_test_set)
    st.write('test sample shape : ')
    st.write(data_test_set.shape)
    #set up model
    problem_type, best_model = dataSetup(data_sample, target_column_name)
    if problem_type:
        #choose model & train
        best_model_trained = compare_models() 
        #test
        test_predict_model(best_model_trained, data_test_set)

