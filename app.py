import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the entire model
emotion_dict = {0: 'Angry',1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6:'Surprise'}


classifier = keras.models.load_model(r'C:\Users\sarab\Desktop\Neural_N_CNN\best_model.h5')

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier(r'C:\Users\sarab\Desktop\Neural_N_CNN\haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Define the VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(
                roi_gray, (56, 56), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No Faces', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activities = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Jenan Sarabi    
            [LinkedIn](https://www.linkedin.com/in/jenan-sarabi/)"""
    )

    if choice == "Home":
        # Add code for the Home section
        pass
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(
            key="example", video_transformer_factory=VideoTransformer
        )

    elif choice == "About":
        # Add code for the About section
        pass

if __name__ == "__main__":
    main()
