import streamlit as st
import numpy as np
from tensorflow.keras import models
import cv2
from PIL import Image 

def image_preproccess(file):
    img = Image.open(file)
    img_array = np.asarray(img, dtype='uint8')
    img_array = cv2.resize(img_array, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return np.array(img_array)

def predict_outcome(image_path): 
    image = image_preproccess(image_path)
    yhat = model.predict(np.array( [image,] ) )
    return yhat
    
 
if __name__=='__main__':
    model = models.load_model('/home/xrusa/Documents/euclidean-eukalyptus/work_in_progress/Final project/web_app/model/model_endometriosis.h5')
    cover = Image.open('/home/xrusa/Downloads/endometriosis_cover.png')
    st.title("Classification of laparoscopic images - Endometriosis")
    st.image(cover, caption = 'Source: ncdalliance.org')    

    uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg','jpg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        yhat = predict_outcome(uploaded_file)
        if  np.argmax(yhat, axis=1) == 1:
            st.write(
                f"The model indicates that this is a pathological picture with {np.round(np.max(yhat)*100)}% accuracy"
                )
        else:
            st.write(
                f"This appears to be an image without pathological signs of endometriosis with {np.round(np.max(yhat)*100)}% accuracy"
                )



