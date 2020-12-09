import streamlit as st
from PIL import Image
import numpy as np
import cv2

from preprocess import detect_face
from preprocess import resize_256
from face2mask import face2mask
from preprocess import resize_origin
from preprocess import return_face



st.title('face2mask')
st.text('画像上の人物の顔にマスクを付与するアプリです。（複数人でも可）\n')
st.text('※画像が表示されない場合は、右上のメニューからRerunしてください。')

uploaded_file = st.file_uploader('マスクを付与したい画像を選択してください。', type=['jpg','png','jpeg'])
if uploaded_file is not None:
    img = Image.open(uploaded_file) #PIL
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #cv2
    
    imgs_face, positions_face = detect_face(img) #detect and cut face
    if imgs_face == None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption='Face is NOT found.', use_column_width=True)
    else:
        imgs_face = resize_256(imgs_face) #resize to 256
        imgs_face = face2mask(imgs_face) #face2mask
        imgs_face = resize_origin(imgs_face, positions_face) #resize to original size
        img = return_face(img, imgs_face, positions_face) #return face to img
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        st.image(img, caption='Put on masks.', use_column_width=True)