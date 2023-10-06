import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Define the number of latent variables
num_latent_vars = 16

import streamlit as st

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
    # Reconstruct data
    latent_layer = encoder.predict(data_file[:, :512, :].astype('float32'))
    return latent_layer

# Main code
def main():
    st.title("Original vs Reconstructed")

    # Introduction
    st.header("Introduction")
    st.markdown("""
    To gain a more profound understanding of how well the Variational Autoencoder (VAE) is able to capture the signal in 12 latent variables, 
    we created the possibility to process a randomly selected epoch of gait using the VAE. 
    On the top panel is the original signal, on the bottom panel is the reconstructed signal. 
    The outcomes of the latent features are described in the table below.
    """)

    # Panel for Original vs Reconstructed
    st.subheader("Original vs Reconstructed Signal Visualization")
    st.markdown("On the top panel is the original signal, and on the bottom panel is the reconstructed signal.")



    st.divider()
    # Load models
    encoder, decoder = load_models()

    # Load data
    data_np = load_data()

    epochLength = 512
    
    example_chart_placeholder = st.empty()
    st.divider()

    # Create an empty placeholder for the reconstructed line chart
    reconstructed_chart_placeholder = st.empty()

    st.divider()
    
    st.subheader("Latent Layer Visualization")
    latent_chart_placeholder = st.empty()
    
    with st.spinner("Loading data..."):
        file_number = np.random.randint(data_np.shape[0])
        data_file = data_np[file_number, :epochLength, :]
        example_chart_placeholder.line_chart(data_file)
        latent_layer = encode_data(encoder, data_file, epochLength)

        # Display the latent layer as a bar chart

        latent_chart_placeholder.bar_chart(latent_layer[0])

        reconstructed_data = reconstruct_data(decoder, latent_layer)
        # Update the placeholder with the reconstructed line chart
        reconstructed_chart_placeholder.line_chart(reconstructed_data[0, :epochLength, :])

    # Button to reload a different example
    if st.button('Load different example'):
        with st.spinner("Loading data..."):
            
            example_chart_placeholder.empty()
            reconstructed_chart_placeholder.empty()
            latent_chart_placeholder.empty()
            
            file_number = np.random.randint(data_np.shape[0])
            data_file = data_np[file_number, :epochLength, :]
            example_chart_placeholder.line_chart(data_file)
            latent_layer = encode_data(encoder, data_file, epochLength)

            

            reconstructed_data = reconstruct_data(decoder, latent_layer)
            # Update the placeholder with the reconstructed line chart
            reconstructed_chart_placeholder.line_chart(reconstructed_data[0, :epochLength, :])
            # Update the latent layer bar chart
            st.bar_chart(latent_layer[0])

if __name__ == '__main__':
    main()
