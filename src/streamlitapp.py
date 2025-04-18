# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
import subprocess  # Better handling of system commands
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the s1 folder inside data (relative to current script directory)
data_path = os.path.join(BASE_DIR, 'data', 's1')

# List the files in the s1 folder
options = os.listdir(data_path)

selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')

        # Build the correct file path for the selected video
        file_path = os.path.join(data_path, selected_video)

        # Use ffmpeg to convert video to mp4 (ensure correct file handling)
        output_video = 'test_video.mp4'
        subprocess.run(f'ffmpeg -i "{file_path}" -vcodec libx264 {output_video} -y', shell=True, check=True)

        # Rendering inside the app
        try:
            with open(output_video, 'rb') as video:
                video_bytes = video.read() 
                st.video(video_bytes)
        except FileNotFoundError as e:
            st.error(f"Error: {e}")

    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')

        # Load the video data
        try:
            video, annotations = load_data(tf.convert_to_tensor(file_path))

            # Save video as gif for display
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')

            # Load the model and make prediction
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
        
        except Exception as e:
            st.error(f"Error processing video: {e}")
        
        # Clean up the temporary files
        if os.path.exists(output_video):
            os.remove(output_video)
        if os.path.exists('animation.gif'):
            os.remove('animation.gif')
