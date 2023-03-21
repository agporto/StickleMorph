import cv2
import dlib
import glob
import numpy as np
import pandas as pd
import PIL
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_javascript import st_javascript



PAGE_CONFIG = {"page_title": "StickleMorph", "page_icon": ":o", "layout": "wide"}
st.set_page_config(**PAGE_CONFIG)

def clear_cache():
    st.session_state.landmarks = False
    st.session_state.initial = []

def format_labels(predictor):
    """Format predictor labels."""
    return predictor.split("/")[-1].split(".")[0]
    
def download_csv(session):
    """Download landmarks data as a CSV."""
    if session['landmarks'] is False:
        session['landmarks'] = session['initial']
    df = pd.DataFrame(session['landmarks'], columns=['x', 'y'])
    return df.to_csv(index=False).encode("utf-8")

def landmarks_to_fabric_circles(landmarks, point_color, text_color):
    """Convert landmarks to fabric circles."""
    circles_and_numbers = []
    parts = map(str, range(0, len(landmarks)))
    ids = sorted(list(parts))

    for i, (x, y) in enumerate(landmarks):
        circles_and_numbers.append({
            "type": "circle",
            "left": x - 5,
            "top": y - 5,
            "radius": 5,
            "fill": point_color,
            "selectable": True,
        })
        circles_and_numbers.append({
            "type": "text",
            "text": str(ids[i]),
            "left": x + 10,
            "top": y + 10,
            "fontSize": 16,
            "fill": text_color,
            "selectable": False,
        })

    return {"objects": circles_and_numbers, "background": "rgba(0, 0, 0, 0)"}

# Set up session state
if 'landmarks' not in st.session_state:
    st.session_state.landmarks = False
if 'initial' not in st.session_state:
    st.session_state.initial = []

inner_width = st_javascript("""window.innerWidth""")
# Load shape predictor models
predictors = glob.glob("predictors/*.dat")

# Sidebar
st.sidebar.image("resources/logo_v4.png", use_column_width=True)

st.sidebar.markdown("## Predictors")
selected_model = st.sidebar.selectbox("Choose a predictor model", options=predictors, format_func=format_labels, on_change=clear_cache)

st.sidebar.markdown("## Image Dimensions")
maximum = st.sidebar.slider("Maximum width", min_value=200, max_value=2000, value=1000, step=50)

st.sidebar.markdown("## Edit Landmarks")
edit = st.sidebar.radio("Choose:", options=["Locked", "Editable"], label_visibility="collapsed")
cola, colb = st.sidebar.columns(2)

submit = cola.button("Submit Edits")
clear = colb.button("Clear Edits")

st.sidebar.markdown("## Filter Output")
filter = st.sidebar.multiselect("Choose landmarks to remove from output", options=range(0, 68))


st.sidebar.markdown("## Color")
cola,colb = st.sidebar.columns(2)
landmark_color = cola.color_picker("Landmark: ", "#00ff00")
stroke_color = colb.color_picker("Text: ", "#ffffff")



# Main area
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image", on_change = clear_cache)

if uploaded_file is not None:
    # Main area
    if inner_width < maximum:
        maximum = inner_width

    img_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ratio = maximum / w if w != 0 else 1

    rect = dlib.rectangle(1, 1, int(image.shape[1]) - 1, int(image.shape[0]) - 1)
    predictor = dlib.shape_predictor(selected_model)
    shape = predictor(image,rect)
    landmarks = [(point.x, point.y) for point in shape.parts()]
    st.session_state.initial = landmarks.copy()

    # Create a canvas to draw and update landmarks
    if edit == 'Locked':
        if clear:
            st.session_state.landmarks = st.session_state.initial
        if st.session_state.landmarks is not False:
            landmarks = st.session_state.landmarks
        parts = map(str, range(0, len(landmarks)))
        ids = sorted(list(parts))
        deviation = int(10*(1/ratio))
        for i, (x, y) in enumerate(landmarks):
            image = cv2.circle(image, (x, y), int(7/ratio), PIL.ImageColor.getcolor(landmark_color,'RGB'), -1)
            image = cv2.putText(image, str((ids[i])), (x+deviation, y+deviation*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5/ratio, PIL.ImageColor.getcolor(stroke_color,'RGB'), int(0.8/ratio), cv2.LINE_AA)
        st.image(image, width=maximum, clamp=True)
        if clear:
            st.session_state.landmarks = st.session_state.initial
        
    else:
        if clear:
            st.session_state.landmarks = st.session_state.initial
        if st.session_state['landmarks'] is not False:
            landmarks = st.session_state['landmarks']

        
        landmarks = [(int(x*ratio), int(y*ratio)) for x, y in landmarks]
        image = cv2.resize(image, (int(w*ratio), int(h*ratio)))

        canvas_result = st_canvas(
            background_image=PIL.Image.fromarray(image),
            stroke_width=3,
            stroke_color= stroke_color,
            background_color="rgba(0, 0, 0, 0)",
            update_streamlit=True,
            height=image.shape[0],
            width=image.shape[1],
            drawing_mode=['transform'],
            initial_drawing=landmarks_to_fabric_circles(landmarks, landmark_color, stroke_color),
            key="canvas",
        )


        objects = pd.json_normalize(canvas_result.json_data["objects"])

        circles = objects[objects['type'] == 'circle']
        texts = objects[objects['type'] == 'text']

        # Get the text objects immediately below each circle object
        circles.loc[:, 'landmark_id'] = texts['text'].values

        # Create a new dataframe with the required columns
        final_df = circles[['landmark_id', 'left', 'top']]
        final_df.columns = ['landmark_id', 'x', 'y']
        final_df.landmark_id = final_df.landmark_id.astype(int)
        #final_df = final_df.sort_values(by='landmark_id').reset_index(drop=True)

        if submit:
            final_df['x'] = final_df['x'] + 5
            final_df['y'] = final_df['y'] + 5
            st.session_state.landmarks = [(int(x*1/ratio), int(y*1/ratio)) for x, y in final_df[['x', 'y']].values]

    st.sidebar.markdown("## Download")
    colc,cold = st.sidebar.columns(2)
    colc.download_button(
                label="Download TPS",
                data=download_csv(st.session_state),
                file_name="landmarks.tps",
                mime="text/tps",
                use_container_width=True,
                #on_click= download_tps,
    )
    cold.download_button(
                label="Download CSV",
                data=download_csv(st.session_state),
                file_name="landmarks.csv",
                mime="text/csv",
                use_container_width=True,
                #on_click= download_csv,
    )


