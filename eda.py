import streamlit as st
import pandas as pd
import numpy as np


def run_eda():
    st.subheader("Exploratory Data Analysis")
    df = pd.read_csv(r"data\diabetes_data_upload.csv")
    st.dataframe(df)