# Imports
import streamlit as st
import tensorflow as tf
import os
import numpy as np

CLASS_NAMES = ["Cat", "Dog"]

## Page Title
st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title(" Cat vs Dogs Image Classification")
st.markdown("---")

## Select & Initialize TFLite intepreter
st.sidebar.header("TF Lite Models")
display = ("Converted FP-16 Quantized Model", "Converted Integer Quantized Model", "Converted Dynamic Range Quantized Model")
options = list(range(len(display)))
value = st.sidebar.selectbox("Select Model", options, format_func=lambda x: display[x])
st.write(value)

if value == 0:
    tflite_interpreter = tf.lite.Interpreter(model_path='model\model_fp16.tflite')
    tflite_interpreter.allocate_tensors()
if value == 1:
    tflite_interpreter = tf.lite.Interpreter(model_path='model\model_int8.tflite')
    tflite_interpreter.allocate_tensors()
if value == 2:
    tflite_interpreter = tf.lite.Interpreter(model_path='model\model_dynamic.tflite')
    tflite_interpreter.allocate_tensors()

## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=['jpg', 'png', 'jpeg', 'bmp'])

if uploaded_file is not None:
    # Copy image
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocessing
    path = os.path.join("tempDir", uploaded_file.name)
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    if value == 1 or value == 2:
        img = tf.image.convert_image_dtype(img, tf.uint8)

    # Image => Tensor
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    st.image(img)

## Run inference
if st.button("Get Predictions"):
    tensor_index = tflite_interpreter.get_input_details()[0]['index']
    input_tensor = tflite_interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = img_array
    
    tflite_interpreter.invoke()

    output_details = tflite_interpreter.get_output_details()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)

    st.success(CLASS_NAMES[tflite_model_prediction])
