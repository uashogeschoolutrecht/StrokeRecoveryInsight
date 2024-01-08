import streamlit as st


def add_home_button():
    # Use local CSS to style the button if needed
   st.sidebar.markdown(f'<a href="https://strokerecovery.makingsenseofse.src.surf-hosted.nl/" target="_self"><button class="btn-primary">Back to the homepage</button></a>', unsafe_allow_html=True)

    # ... add other sidebar elements here ...

# Add the home button first
add_home_button()
# Page title
st.title("Exploring Unsupervised Feature Extraction of IMU-Based Gait Data in Stroke Rehabilitation using a Variational Autoencoder")

# Introduction
st.header("Introduction")
st.markdown("""
To gain a more comprehensive understanding of gait recovery, monitor progress, and tailor interventions, measuring the way people walk is crucial [1,2,3]. 
Inertial Measurement Units (IMUs) are small and portable sensors that enable objective and continuous measurements of the way people walk. However, IMU data needs to be processed to extract relevant information before it can be used in research and clinical practice.

This study explored a data-driven approach of processing IMU data using Variational AutoEncoder (VAE) [4]. 
A VAE is a generative model that employs deep learning techniques to learn a compact, low-dimensional representation of data.
""")

# Variational AutoEncoder
st.header("Variational AutoEncoder")
st.markdown("""
The VAE comprises two main components: an encoder and a decoder.
- The encoder maps the input data to a lower-dimensional representation, known as the latent layer, by encoding it into a mean and variance vector.
- The decoder takes this sample as input and generates a reconstructed output that is similar to the original input data.

The input and output of the VAE consisted of a 512X6 epoch. The encoder and decoder both comprised three convolutional layers. 
The latent layer contained 12 latent variables.

![Figure 1: Variational autoencoder (VAE) used in this study.](https://i.ibb.co/mykQWsb/Picture-1.png)

*Figure 1: Variational autoencoder (VAE) used in this study. The input was a 512X6 epoch of an IMU-based gait measurement. 
The encoder (green) and the decoder (blue) consisted of 3 mirrored convolutional layers with a size of 256, 128, and 64 nodes. 
These layers were configured with 32, 64, and 128 filters, respectively, and employed a kernel size of 3. 
The activation function used throughout the model was tanh. The latent layer contained 12 normally distributed latent features. 
The model was trained by comparing the input to the reconstructed output. A tanh activation function was used in the convolutional layers. 
An Adam optimizer with a learning rate of 0.001 was used.*

## References
[1] Sung Shin, Robert Lee, Patrick Spicer, and James Sulzer. Does kinematic gait quality improve with functional gait recovery? 
    Journal of Biomechanics, 105:109761, 03 2020.

[2] Elizabeth Wonsetler and Mark Bowden. A systematic review of mechanisms of gait speed change post-stroke. 
    Topics in Stroke Rehabilitation, 24:1–12, 02 2017.

[3] Michiel Punt, Sjoerd Bruijn, Kim van Schooten, Mirjam Pijnappels, Ingrid Port, Harriet Wittink, and Jaap Van Dieen. 
    Characteristics of daily life gait in fall and non fall-prone stroke survivors and controls. 
    Journal of NeuroEngineering and Rehabilitation, 13, 07 2016.

[4] Diederik Kingma and Max Welling. An Introduction to Variational Autoencoders. 01 2019.
""")
