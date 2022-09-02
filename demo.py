from helpers import build_model, get_results

import cv2
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO


def process_image(input_image):
    img = Image.fromarray(input_image)
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    return buffer.getvalue()


st.set_page_config(page_title="Bounding Box",page_icon="☯️")

if 'THRESH' not in st.session_state: # Since model is always dependent on threshold
    st.session_state['THRESH'] = 0.6
    st.session_state['model'] = build_model(st.session_state['THRESH'])

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = None

if 'result' not in st.session_state: # download can only exists iff there's result
    st.session_state['result'] = None
    st.session_state['download'] = None

  
def slider_callback():
    st.session_state['model'] = build_model(st.session_state['THRESH'])


with st.sidebar:
    st.markdown("""## Change Confidence Threshold""")
    THRESH = st.slider('', key = 'THRESH',
    min_value = 0.1, max_value = 0.99, step = 0.05,
    help = "Enter a number between 0.1-.99: Detections are filtered on this confidence",
    on_change = slider_callback) # changing this will create model again with a new threshold"
        

st.markdown("## Upload Image")

st.session_state['uploaded'] = st.file_uploader("", type = ["jpg", "png", "jpeg"], key = "upload", help = "Upload the Image from your device")

if st.session_state['uploaded'] is not None:
    file_bytes = np.asarray(bytearray(st.session_state['uploaded'].read()), dtype=np.uint8) # open with OpenCv
    st.session_state['uploaded'] = cv2.imdecode(file_bytes, 1) # change session state 

    with st.spinner("Processing Image..."):
        st.session_state['result'] = get_results(st.session_state['model'], st.session_state['uploaded']) # Get Bounding Boxes

    st.markdown(f"""<b> Current Threshold: <span style="color: #ff0000">{str(st.session_state['THRESH'])} </span> </b>|| Change threshold to see variations in results""",
    unsafe_allow_html=True) # Show current threshold value

    st.session_state['download'] = process_image(st.session_state['result'])
    _ = st.download_button(label="Download",data = st.session_state['download'], file_name="image.jpeg",mime="image/jpeg")
    
    st.image(st.session_state['result'], channels="RGB")



