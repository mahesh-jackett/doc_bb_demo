from helpers import *
import streamlit as st



st.set_page_config(page_title="BB Model",page_icon="☯️", layout="wide")

if 'THRESH' not in st.session_state: # Since model is always dependent on threshold
    st.session_state['THRESH'] = 0.6
    st.session_state['model'] = TorchModel(st.session_state['THRESH'])

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = None

if 'base_result' not in st.session_state:
    st.session_state['base_result'] = None
    st.session_state['eigen'] = None
    st.session_state['norm_eigen'] = None
    st.session_state['meta'] = None

  
def slider_callback():
    ''''
    Build the modle again and clear the sesssions to get new predictions on new model
    '''
    with st.spinner("Building Model again with new threshold..."):
        st.session_state['model'] = TorchModel(st.session_state['THRESH'])
        upload_callback() # it'll force the predictions to be run again


def upload_callback():
    '''
    Clear the session states so that we can do predictions on the freshly uploaded image
    '''
    st.session_state['base_result'] = None
    st.session_state['eigen'] = None
    st.session_state['norm_eigen'] = None


with st.sidebar:
    st.markdown("""## Change Confidence Threshold to see variations in results""")
    THRESH = st.slider('', key = 'THRESH',
    min_value = 0.1, max_value = 0.99, step = 0.05,
    help = "Enter a number between 0.1-.99: Detections are filtered on this confidence",
    on_change = slider_callback) # changing this will create model again with a new threshold"
        

st.markdown("## Upload Image")
st.session_state['uploaded'] = st.file_uploader("", type = ["jpg", "png", "jpeg"], on_change = upload_callback, help = "Upload the Image from your device")

if st.session_state['uploaded'] is not None:
    try:
        file_bytes = np.asarray(bytearray(st.session_state['uploaded'].read()), dtype=np.uint8) # open with OpenCv
        st.session_state['uploaded'] = cv2.imdecode(file_bytes, 1) # change session state 
    
    except Exception as e:
        st.exception(f"""<b><span style="color: #ff0000">Image Loading Error: </span>{e}</b><br>""", unsafe_allow_html=True)

    try:
        if st.session_state['base_result'] is None:
            with st.spinner("Getting Model Inference..."):
                float_image, tensor_image, boxes, labels, classes, COLORS = get_results(st.session_state['model'], st.session_state['uploaded'])# Get Bounding Boxes
                st.session_state['base_result'] = draw_boxes(st.session_state['uploaded'],boxes, labels, classes, COLORS)
                st.session_state['meta'] = float_image, tensor_image, boxes, labels, classes, COLORS
                st.session_state['show'] = st.session_state['base_result']
        else:
            float_image, tensor_image, boxes, labels, classes, COLORS = st.session_state['meta']

    except Exception as e:
        st.exception(f"""<b><span style="color: #ff0000">Inference Error: </span>{e}</b><br>""", unsafe_allow_html=True)
    

    st.markdown("<h5>Choose the type of results you want to see. Save the image results by clicking 'Save image as..'</h5><br>", unsafe_allow_html = True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Default Predictions"): 
            st.session_state['show'] = st.session_state['base_result']


    with col2:
        if st.button("EigenCam: Box Only"):
            if st.session_state['norm_eigen'] is None:
                with st.spinner("Getting Box Normalized Eigen Cam Results..."):
                    st.session_state['eigen'], st.session_state['norm_eigen'] = eigen_res(st.session_state['model'], float_image, tensor_image, labels, boxes, classes, COLORS)
            
            st.session_state['show'] = st.session_state['norm_eigen']
    

    with col3:
        if st.button("EigenCam: Whole Image",):
            if st.session_state['eigen'] is None:
                with st.spinner("Getting Eigen Cam Results..."):
                    st.session_state['eigen'], st.session_state['norm_eigen'] = eigen_res(st.session_state['model'], float_image, tensor_image, labels, boxes, classes, COLORS)
            
            st.session_state['show'] = st.session_state['eigen']
    
    
    try:
        st.image(st.session_state['show'], channels="RGB")
    except Exception as e:
        st.exception(f"Image Rendering Error: {e}")