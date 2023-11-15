import streamlit as st
import numpy as np
import pandas as pd

# Main code
def main():
    st.title("Processing your own collected data")

    # Introduction
    st.header("Introduction")
    st.markdown("""
    On this page, we enable you to upload your own collected 
    balance data.
    """)

    # Create 5 file uploaders
    uploaded_files = []
    for i in range(5):
        uploaded_file = st.file_uploader(f"Upload File {i+1}", type=["csv"])
        if uploaded_file is not None:
            uploaded_files.append(uploaded_file)

    # Process uploaded files
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            st.header(f"Uploaded File {i+1}")
            data = pd.read_csv(uploaded_file, names=['T', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'Time'], sep=',', skiprows=10)

            # Select only 'Ax', 'Ay', and 'Az' columns
            selected_data = data[['ax', 'ay', 'az', 'gx', 'gy', 'gz']]
            st.line_chart(selected_data)
            st.write('## Selecteer een subset van 512 samples')

            # Create a slider widget for selecting the range
            start_index = st.slider('Start index', 0, len(selected_data) - 512, 0)
            end_index = start_index + 512

            # Get the subset of data based on the selected range
            subset_data = selected_data.iloc[start_index:end_index]

            # Display the selected data
            st.line_chart(subset_data)

if __name__ == '__main__':
    main()
