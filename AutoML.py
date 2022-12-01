import streamlit as st
import pandas as pd
import os

# Libraries for Profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# Libraries for ML
# from pycaret.classification import setup, compare_models, pull, save_model
# from pycaret.regression import setup, compare_models, pull, save_model


with st.sidebar:
    st.title("Auto ML Process")
    st.image("robot.png")
    option = st.radio("Select a Process", ["Input", "Data Exploration", "Model Selection", "Model Download"])
    st.info("AutoML is the process of automating tasks of applying machine learning to real-world problems.")



if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if option == "Input":

    st.title("Select Your File Extension")
    extension = st.selectbox("File Extension", ('CSV', 'Excel', 'Parquet'))
    st.title("Input Data")
    file = st.file_uploader("Please Upload Your Dataset Here !")
    if extension == "CSV":
       if file:
           df = pd.read_csv(file, index_col=None)
           df.to_csv("sourcedata.csv", index=None)
           st.dataframe(df)

    elif extension == "Excel":
        if file:
           df = pd.read_excel(file, index_col=None)
           df.to_excel("sourcedata.xlsx", index=None)
           st.dataframe(df)
        
    elif extension == "Parquet":
        if file:
           df = pd.read_parquet(file, index_col=None)
           df.to_parquet("sourcedata.xlsx", index=None)
           st.dataframe(df)


    

elif option == "Data Exploration":
    st.title("Automated Data Exploration")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

elif option == "Model Selection":

    st.title("Select ML Algorithm")
    algorithm = st.selectbox("Algorithms", ('- Select -', 'Classification', 'Regression'))

    if algorithm == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model
        st.title("Classification Model")
        target = st.selectbox("Select Target Feature", df.columns)
        if st.button("Train Model"):
            setup(df, target=target, silent=True)
            setup_df = pull()
            st.info("This is the ML Experiment Setting")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')

    if algorithm == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model
        st.title("Regression Model")
        target = st.selectbox("Select Target Feature", df.columns)
        if st.button("Train Model"):
            setup(df, target=target, silent=True)
            setup_df = pull()
            st.info("This is the ML Experiment Setting")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'best_model')



    

elif option == "Model Download":
    st.title("Your Model Is Ready !")
    st.image("box.png", width=500)
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model", f, "trained_model.pkl")