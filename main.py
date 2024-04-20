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
    st.write("count plot for categorical columns:")
    for col in categoricalCols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data[col])
        plt.title(f"{col} Count Plot")
        plt.xlabel(col)
        plt.ylabel("Count")
        #show on web app
        st.pyplot()

def train(data, targetColumn):
    setup(data=data, target=targetColumn,train_size=0.75, session_id=10,remove_outliers=True, normalize=True,normalize_method='minmax', transform_target=True)

    if data[targetColumn].dtype == 'object':
        #classification
        problemType = 'classification'
        print('classification problem')
        model_options = ['lr', 'knn', 'ridge', 'lasso', 'dt', 'rf', 'xgboost']
    else:
        #regression
        problemType = 'regression'
        print('regression problem')
        model_options = ['lr', 'knn', 'nb', 'dt', 'rf', 'xgboost']

    #select model
    selected_model = st.selectbox("Select Model", model_options)
    #train selected model
    model = create_model(selected_model)
    #finalize model
    final_model = finalize_model(model)
    #print model
    st.write("model selected:", model)

    return final_model,problemType

def test(model, X_test, y_test,problemType):

    st.subheader("Model Evaluation")

    if problemType == 'classification':
        predictions = predict_model(model, data=X_test)
        st.write("Classification Report:")
        st.write(classification_report(y_test, predictions['Label']))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, predictions['Label']))
    else:
        predictions = predict_model(model, data=X_test)
        st.write("Mean Absolute Error:", mean_absolute_error(y_test, predictions['Label']))
        st.write("Mean Squared Error:", mean_squared_error(y_test, predictions['Label']))
        st.write("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions['Label'])))
        st.write("R-squared:", r2_score(y_test, predictions['Label']))

#getting data
def main():
    print('hello')
    # app title
    st.title("Machine Learning Model Input Form")
    # add a header
    st.header("Input Parameters")
    #prompt the user to upload the dataset
    uploaded_file = st.file_uploader("Upload file")
    #target column
    target_column_name = st.text_input("Enter the name of the target column:", value='targetColumn')
    #add a button to submit the form
    submit_button = st.button("Submit")

    # process input when its Submited
    if submit_button:
        try:
            # upload
            upF = uploaded_file
            df = get_data(upF)
            st.write("Input DataFrame:")
            st.write(df)
            #call the functions
            EDA(df)
            model,problem = train(df,target_column_name)
            X = df.drop(target_column_name, axis=1)
            y = df[target_column_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            test(model, X_test, y_test, problem)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    if __name__ == "__main__":
        main()