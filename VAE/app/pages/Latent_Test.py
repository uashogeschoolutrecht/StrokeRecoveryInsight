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
    encoder = tf.keras.models.load_model('VAE/app/VAE_512_encoder', compile=False)
    decoder = tf.keras.models.load_model('VAE/app/VAE_512_decoder', compile=False)
    return encoder, decoder


# Load data and set cache to prevent reloading
@st.cache_resource
def load_data():
    data_np = np.load('VAE/app/512.npy', allow_pickle=True)
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
    # Reconstruct data
    # print(data_file.shape)
    # print(data_file.astype('float32').shape)
    # data_np[:, :512, :].astype('float32')
    latent_layer = encoder.predict(data_file[:, :512, :].astype('float32'))
    # latent_layer = np.concatenate(
    #     (data_np[:, epochLength:, 0], latent_layer), axis=1)
    return latent_layer
# Create a sidebar
with st.sidebar:
    sliders = []
    for i in range(num_latent_vars):
        slider = st.slider(f'L{i+1}', -5, 5, 0, key=f'slider_{i}')
        sliders.append(slider)

# Main code
def main():
    # Page title
    st.title("Average Signal")

    # Introduction
    st.header("Introduction")
    st.markdown("""
    One disadvantage of using deep-learning techniques in general is the 'black box problem', meaning that it is often very difficult to figure out how the model made the decisions and predicted the outcomes. 
    This is also the case with Variational Autoencoders (VAE), however, VAE are also generative models which means that it is possible to generate non-existing data by filling in the latent features. 
    By changing values of the latent features, it is possible to get insight into which aspect of the raw signal a latent feature represents.
    """)

    # Panel for the average signal
    st.subheader("Average Signal Visualization")
    st.markdown("In the panel below, the average signal (all latent features at 0) is depicted. The sliders below can be used to adjust the value of the latent features.")


    # Load models
    encoder, decoder = load_models()

    # Load data
    data_np = load_data()

    epochLength = 512
    
    # Create an empty placeholder for the reconstructed line chart
    reconstructed_chart_placeholder = st.empty()
    for i in range(num_latent_vars):
            reconstructed_data = reconstruct_data(decoder, sliders)
            reconstructed_chart_placeholder.line_chart(reconstructed_data[0, :epochLength, :])
    
    
if __name__ == '__main__':
    main()
