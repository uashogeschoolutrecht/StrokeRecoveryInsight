import streamlit as st

def add_home_button():
    # Use local CSS to style the button if needed
   st.sidebar.markdown(f'<a href="https://strokerecovery.makingsenseofse.src.surf-hosted.nl/" target="_self"><button class="btn-primary">Back to the homepage</button></a>', unsafe_allow_html=True)

    # ... add other sidebar elements here ...

# Add the home button first
add_home_button()

# Page title
st.title("Gait Assessment")

# Introduction
st.header("Introduction")
st.markdown("""
Balance is often affected after stroke, severely impacting activities of daily life. Conventional testing methods to assess balance provide limited information, as they are subjected to floor and ceiling effects. Instrumented tests, for instance using inertial measurement units, offer a feasible and promising alternative.
""")


st.markdown("""

## References
[1] Felius, Richard & Geerars, Marieke & Bruijn, Sjoerd & Wouda, Natasja & Diee¨n, J.H. & Punt, Michiel. (2022). Reliability of IMU-Based Balance Assessment in Clinical Stroke Rehabilitation. Gait & Posture. 98. 10.1016/j.gaitpost.2022.08.005. 

""")
