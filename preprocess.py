import streamlit as st
import cv2



@st.cache(allow_output_mutation=True)
# detect and cut face
def detect_face(img):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(img_g, minNeighbors=3, scaleFactor=1.1)
    
    if faces == []:
        return None, None
    
    height, width = img.shape[:2]
    imgs_face = []
    positions_face = []
    for face in faces:
        x, y, w, h = face[0], face[1], face[2], face[3]
        ex_w = round(w*0.2)
        x_min = x - ex_w
        x_max = x + w + ex_w
        
        ex_h = round(h*0.2)
        y_min = y - ex_h
        y_max = y + h + ex_h
        
        if x_min < 0: x_min = 0
        if x_max > width: x_max = width
        if y_min < 0: y_min = 0
        if y_max > height: y_max = height
        
        y_min,y_max,x_min,x_max = int(y_min), int(y_max), int(x_min), int(x_max)
        img_face = img[y_min:y_max, x_min:x_max]
        imgs_face.append(img_face)
        positions_face.append([x_min,y_min,x_max,y_max])
    
    return imgs_face, positions_face



# resize to 256
def resize_256(imgs_face):
    imgs_256 = []
    for img_face in imgs_face:
        img_face = cv2.resize(img_face, (256,256))
        imgs_256.append(img_face)
    return imgs_256



# resize to original size
def resize_origin(imgs_face, positions_face):
    imgs_origin = []
    for img_face, position_face in zip(imgs_face, positions_face):
        x_min,y_min,x_max,y_max = position_face
        w = x_max - x_min
        h = y_max - y_min
        img_face = cv2.resize(img_face, (w,h))
        imgs_origin.append(img_face)
    return imgs_origin



# return face to img
def return_face(img, imgs_face, positions_face):
    for img_face, position_face in zip(imgs_face, positions_face):
        height = img_face.shape[0]
        height = int(round(0.4*height))
        img_face = img_face[height:, :]
        x_min,y_min,x_max,y_max = position_face
        h = y_max - y_min
        h = int(round(0.4*h))
        img[y_min+h:y_max, x_min:x_max] = img_face
    return img