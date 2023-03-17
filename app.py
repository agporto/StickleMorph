import streamlit as st
import PIL
import glob
import cv2
import dlib
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas

PAGE_CONFIG = {"page_title": "StickleMorph", "page_icon": ":o", "layout": "wide"}
st.set_page_config(**PAGE_CONFIG)


def landmarks_to_fabric_circles(landmarks, color):
    circles_and_numbers = []
    parts = map(str, range(0, len(landmarks)))
    ids = sorted(list(parts))

    for i, (x, y) in enumerate(landmarks):
        circles_and_numbers.append({
            "type": "circle",
            "left": x - 2,
            "top": y - 2,
            "radius": 5,
            "fill": color,
            "selectable": True,
        })
        circles_and_numbers.append({
            "type": "text",
            "text": str(ids[i]),
            "left": x + 10,
            "top": y + 10,
            "fontSize": 16,
            "fill": "#ffffff",
            "selectable": False,
        })

    return {"objects": circles_and_numbers, "background": "rgba(0, 0, 0, 0)"}


def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    return image


# Load shape predictor models
predictors = glob.glob("predictors/*.dat")

# Sidebar
st.sidebar.image("resources/logo.png", use_column_width=True)
selected_model = st.sidebar.selectbox("Choose a predictor model", options=predictors)

# Main area
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dim = 1400
    image = resize_image(image, max_dim, max_dim)

    rect = dlib.rectangle(1, 1, int(image.shape[1]) - 1, int(image.shape[0]) - 1)
    predictor = dlib.shape_predictor(selected_model)
    shape = predictor(image,rect)
    landmarks = [(point.x, point.y) for point in shape.parts()]

    #drawing_mode = st.sidebar.selectbox("Drawing tool:", ["transform", 'line'])
    stroke_color = st.sidebar.color_picker("Landmark color: ", "#00ff00")

    # Create a canvas to draw and update landmarks
    canvas_result = st_canvas(
        background_image=PIL.Image.fromarray(image),
        stroke_width=3,
        stroke_color= stroke_color,
        background_color="rgba(0, 0, 0, 0)",
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode=['transform'],#drawing_mode,
        initial_drawing=landmarks_to_fabric_circles(landmarks, stroke_color),
        key="canvas",
    )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])

        # Filter only circle objects
        circles = objects[objects['type'] == 'circle']
        texts = objects[objects['type'] == 'text']

        # Get the text objects immediately below each circle object
        circles['landmark_id'] = texts['text'].values

        # Create a new dataframe with the required columns
        final_df = circles[['landmark_id', 'left', 'top']]
        final_df.columns = ['landmark_id', 'x', 'y']
        final_df.landmark_id = final_df.landmark_id.astype(int)
        final_df = final_df.sort_values(by='landmark_id').reset_index(drop=True)

        st.dataframe(final_df)

