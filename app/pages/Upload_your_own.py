
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Define the number of latent variables
num_latent_vars = 16

import streamlit as st

# Add more sliders if needed for additional latent features

# Display the average signal visualization based on the selected latent feature values
# Your visualization code goes here

# Add any additional information or explanations as needed


# Load models and set cache to prevent reloading
@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model('app/VAE_512_encoder', compile=False)
    decoder = tf.keras.models.load_model('app/VAE_512_decoder', compile=False)
    return encoder, decoder

# Load data and set cache to prevent reloading
@st.cache_resource
def load_data():
    data_np = np.load('app/512.npy', allow_pickle=True)
    return data_np


# Reconstruct data
def reconstruct_data(decoder, latent_vars):
    # Create dummy latent variable layer
    latent_layer = np.zeros((1, num_latent_vars))
    
    # Assign slider values to the latent layer
    latent_layer[0] = np.array(latent_vars)
    
    # Reconstruct data
    reconstructed_data = decoder.predict(latent_layer.astype('float32'))
    return reconstructed_data

def encode_data(encoder, data_file, epochLength):
    data_file = np.expand_dims(data_file, axis = 0)
    latent_layer = encoder.predict(data_file[:, :512, :].astype('float32'))
    return latent_layer

# Main code
def main():
    st.title("Processing your own collected data")

    # Introduction
    st.header("Introduction")
    st.markdown("""
    On this page, we enabled you to upload your own collected 
    gait data to evaluate how well the Variational autoencoder can 
    reconstruct your data, and the corresponding outcome of the latent features. 
    Your data should be gait measured with an Inertial Measurement Units (IMUs) located at the foot, stored in format nX6, 
    where n is at least 512 samples long. The uploaded data will not be stored.
    """)

    # Load models
    encoder, decoder = load_models()

    # Load data
    data_np = load_data()

    epochLength = 512
    
    # Create an empty placeholder for the reconstructed line chart
    reconstructed_chart_placeholder = st.empty()
    # for i in range(num_latent_vars):
    #         reconstructed_data = reconstruct_data(decoder, sliders)
    #         reconstructed_chart_placeholder.line_chart(reconstructed_data[0, :epochLength, :])
    st.divider()
    

            
    uploaded_file = st.file_uploader("Upload hier je eigen bestand om te analyseren")
    if uploaded_file is not None:
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

        # Encode the selected data using the encoder
        latent_layer = encode_data(encoder, subset_data.to_numpy(), epochLength)

        # Display the latent variables as a bar chart
        st.subheader("Latent Layer Visualization")
        st.bar_chart(latent_layer[0])

        
if __name__ == '__main__':
    main()
