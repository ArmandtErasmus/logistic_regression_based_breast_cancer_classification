# Import libraries
import streamlit as st 
import pickle as pickle
import pandas as pd



def main():
    
    # Set Streamlit page configuration
    st.set_page_config(
        page_title = 'Breast Cancer Classification with Logistic Regression',
        page_icon = ':female-doctor:',
    )

    st.write('Hello World')


if __name__ == '__main__':
    main()
