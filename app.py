import streamlit as st

from eda import run_eda
from ml import run_ml



def main():
    st.title("Main Application")

    # Menu Setting
    menu = ['Home', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    # if confiction to manage sections
    if choice == "Home":
        st.subheader("Home")

    elif choice == "EDA":
        run_eda()

    elif choice == "ML":
        run_ml()

    elif choice == "About":
        st.subheader("About")



if __name__ == '__main__':
    main()