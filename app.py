import streamlit as st
import streamlit.components.v1 as stc
from eda import run_eda
from ml import run_ml

html_temp = """
		<div style="background-color:white;padding:12px;border-radius:5px">
		<h1 style="color:black;text-align:center;">Early Stage DM Risk Data App </h1>
		<h4 style="color:black;text-align:center;">Diabetes </h4>
		</div>
		"""


def main():
    st.title("Main Application")
    stc.html(html_temp)

    # Menu Setting
    menu = ['Home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    # if condition to manage sections
    # Home section contains basic information about the application
    if choice == "Home":
        st.subheader("Home")
        st.markdown('---')
        st.write("""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Data Source:
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### Application Content:
				- Exploratory Data Analaysis (EDA)
				- Machine Learning Model
			    """)

    # Exploratry data analysis includes plots and some statistical
    # information taken from dataset.
    elif choice == "EDA":
        run_eda()

    # ML contain machine learning models
    elif choice == "ML":
        run_ml()

    # About this programm
    elif choice == "About":
        st.subheader("About")



if __name__ == '__main__':
    main()