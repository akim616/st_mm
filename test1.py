import re
import streamlit as st
from mmocr.utils.ocr import MMOCR
from PIL import Image
import numpy as np
import cv2


def img_upload():
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg", 'png'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, 1)



header=st.container()
data=st.container()
result=st.container()


with header:
    st.title('Business Card OCR')
    st.text('Optical Character Recognition with OpenCV and Pytesseract.')


with data:
    st.header('Upload file')
    image=img_upload()
    if image is not None:
        st.image(image, channels="BGR")


with result:
    st.header('Results:')
    if image is not None:
        ocr = MMOCR(det='TextSnake', recog='Tesseract')
        results = ocr.readtext(image, print_result=False)
        result = results[0]
        st.write(result)
