import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# Functions
@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df


def run_eda():
    st.subheader("Exploratory Data Analysis")

    # Load dataset for exploratory data analysis
    df = load_data(r"data\diabetes_data_upload.csv")
    df_encoded = load_data(r"data\diabetes_data_upload_clean.csv")
    df_freq = load_data(r"data\freqdist_of_age_data.csv")

    # Sub menu to manage type of eda
    sub_menu = st.sidebar.selectbox("EDA Type", ['Descriptive', 'Plots'])
    
    if sub_menu == 'Descriptive':
        st.subheader("Descriptive Data Analysis")
        with st.expander("Loaded Dataframes"):
            # Original Dataframe has categorical variables.
            st.info("Original DataFrame:")
            st.dataframe(df)
            
            st.markdown('---')

            # Cleaned DataFrame does not includes categorical dataframe.
            # This dataframe is cleaned and categorical variables are encoded.
            st.info("Cleaned DataFrame:")
            st.dataframe(df_encoded)
            
            st.markdown('---')

            st.info("Freq Distribution:")
            st.dataframe(df_freq, use_container_width=True)

        # Check dataset's data type
        with st.expander("Data Types"):
            st.dataframe(df.dtypes, use_container_width=True)

        # Statistical summary
        with st.expander("Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)

        # Variable's Distribution
        with st.expander("Class Distributions"):
            st.dataframe(df['class'].value_counts(), use_container_width=True)
        
        # Statistical summary
        with st.expander("Gender Distribution"):
            st.dataframe(df['Gender'].value_counts(), use_container_width=True)


    elif sub_menu == 'Plots':
        st.subheader("Plots")

        # Desing Columns and layouts
        col1, col2 = st.columns([2, 1])

        with col1:
            # Coutplot for Gender variables
            with st.expander("Distribution Plot Of Gender"):
                # Using seaborn to plot
                fig = plt.figure()
                sns.countplot(x=df['Gender'], palette='dark')
                st.pyplot(fig)

                gen_df = df['Gender'].value_counts().to_frame()

                # Creating frequency table of gender
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender", "Counts"]

                # Pie Chart of Gender using plotly express
                p1 = px.pie(gen_df, names='Gender', values = 'Counts')
                st.plotly_chart(p1, use_container_width=True)

            # Coutplot and pie plot of Class
            with st.expander("Distribution Plot Of Class"):
                fig = plt.figure()
                sns.countplot(x=df['class'], palette='dark')
                st.pyplot(fig)

                class_df = df['class'].value_counts().to_frame()

                # Creating frequency table of gender
                class_df = class_df.reset_index()
                class_df.columns = ["class", "Counts"]

                # Pie Chart of Gender using plotly express
                p1 = px.pie(class_df, names='class', values = 'Counts')
                st.plotly_chart(p1, use_container_width=True)

        with col2:
            # Gender Frequncy Table
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df, use_container_width=True)
            
            # Class Frequncy Table
            with st.expander("Class Distribution"):
                st.dataframe(class_df, use_container_width=True)


        st.markdown('---') # Seperats Plots of single variables with frequncy and outlier section

        # Freq Dist
        with st.expander("Frequency Dist Of Age"):

            p2 = px.bar(df_freq, 
                        x = 'Age', 
                        y='count')
            st.plotly_chart(p2)
            
        # Check Age's Outlier
        with st.expander("Outlier Detection"):
            # Outlier Detection
            p3 = px.box(df,
                        x="Age",
                        color="Gender")
            st.plotly_chart(p3)

        # Plot correlation heampat of df_encoded 
        with st.expander("Correlation Plot"):
            corr_mat = df_encoded.corr()
            fig = plt.figure(figsize=(20, 10))
            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm')
            st.pyplot(fig)
