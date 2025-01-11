import streamlit as st
from PIL import Image
import cv2
import time
import dlib
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter('ignore')
from faceswap_cam import face_swap
from detection import *
face_detector = Face_Detector()
lmk_detector = Landmark_Detector()
from tensorflow.keras.applications import resnet50
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
import pickle
from skimage.transform import resize
from skimage.color import rgb2gray
from gtts import gTTS
from playsound import playsound

# page config
st.set_page_config(
    page_title="DeepFake",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

def speak(text):
    tts = gTTS(text, lang='en')

    file_path = os.path.abspath("output.mp3")
    tts.save(file_path)
    try:
        playsound(file_path)
    except Exception as e:
        print("Error playing sound:", e)
    finally:
        os.remove(file_path)

    #tts.save("output.mp3")
    #playsound("output.mp3")
    #os.remove("output.mp3")

filename = 'finalized_model.sav'
# load the model from disk
model = pickle.load(open(filename, 'rb'))

# download  model

@st.cache_resource   #to initialize the model only once
def load_model():
    return resnet50.ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet')

rnet_bmodel = load_model()
# resenet
rnet_model = Model(inputs=rnet_bmodel.input,outputs=rnet_bmodel.layers[-1].output)

def feature_extraction(inp):
  inp1 = np.expand_dims(inp,axis=0)
  pca = PCA(n_components=30)
  fet2 = rnet_model.predict(inp1)
  fet2 = np.array(fet2).flatten().reshape(49,2048)
  fet = pca.fit_transform(fet2).flatten()
  return fet


#Load pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Load the source image
# source_image = cv2.imread("image1.jpg")
# source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
# source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

# Function to extract facial landmarks
#def get_landmarks(image):
#    faces = detector(image)
#    if len(faces) == 0:
#        return None
#    return np.matrix([[p.x, p.y] for p in predictor(image, faces[0]).parts()])

# Function to warp the source face onto the target face
#def warp_face(source_image, source_landmarks, target_image, target_landmarks):
    # Calculate affine transform matrix
#    transform_matrix = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks)[0]
    # Warp the source image onto the target image
#    warped_face = cv2.warpAffine(source_image, transform_matrix, (target_image.shape[1], target_image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#    return warped_face


# load images
top_image = Image.open('static/banner_top.png')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')
# title
st.image(main_image,use_container_width='auto')
st.title('Realtime DeepFake Detection 👨‍🎓')
st.sidebar.image(top_image,use_container_width='auto')
st.sidebar.header('Input 🛠')
## Select camera to feed the model
available_cameras = {'Camera 1': 1, 'Camera 2': 1, 'Camera 3': 1}
cam_id = st.sidebar.selectbox(
    "Select which camera signal to use", list(available_cameras.keys()))
st.sidebar.image(bottom_image,use_container_width='auto')

# checkboxes
st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
col1, col2 ,col3 ,col4  = st.columns([1,6,6,1],gap='large')
with col2:
    live_feed = st.checkbox('Start Web-Camera ✅')
with col3:
    deep_btn = st.checkbox('DeepFake ✅') 
# camera section    
col1, col2 ,col3 = st.columns([1,5,1])
with col2:
    frame_placeholder = st.image('static/live-1.png')





# FS = face_swap("mohan.jpg")
speechflag=1
# # Initialize webcam
cap = cv2.VideoCapture(available_cameras[cam_id])
# # if deep_btn:
flag = 1
# # else:
# #     flag = 0
while cap.isOpened() and live_feed:
    speechflag = speechflag+1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (400,300))
#     swapped_face = frame
#     bboxes, _ = face_detector.detect(frame)  # get faces
    
#     if len(bboxes) != 0:
#         bbox = bboxes[0] # get the first 
#         bbox = bbox.astype(np.int_)
#         lmks, PRY_3d = lmk_detector.detect(frame, bbox)  # get landmarks
#         lmks = lmks.astype(np.int_)
#         try:
#             swapped_face = FS.run(frame,lmks)
#         except:
#             pass
    
#         if flag:
#             gray = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2GRAY)
#         else:
    swapped_face = frame
    gray = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2GRAY)
#         # Detect faces in the grayscale image
#         faces = detector(gray)
#         swapped_face  = cv2.cvtColor(swapped_face , cv2.COLOR_BGR2RGB)
        # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(swapped_face,(x,y),(x+w,y+h),(255,0,0),4)
        face = swapped_face[y:y+h,x:x+w,:]
        face = resize(face,(224,224))
        f = feature_extraction(face)
        if flag:
            result = model.predict([f])
        else:
            prob = model.predict_proba([f])[0]
            print(prob)
            result=[]
            if prob[0]>0.86:
                result.append(1)
            else:
                result.append(0)
        print(result)
        if result[0]:
            cv2.putText(swapped_face,'deep fake detected',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
            if speechflag>10:
                speechflag = 0
                speak('chances for deep fake')
                
        else:
            cv2.putText(swapped_face,'normal',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
             # frame_placeholder.image(face)
            frame_placeholder.image(swapped_face)


    # Display the result
        frame_placeholder.image(swapped_face)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image( frame)

    # Exit when 'q' is pressed
    if (cv2.waitKey(1) & 0xFF ==ord("q")) or not(live_feed):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()


st.markdown("<br><hr><center>Made by PROJECT 9 </center><hr>", unsafe_allow_html=True)