import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
print(tf.__version__)

# Load the trained model
model = tf.keras.models.load_model('Image-Classifier/image_classifier_model.h5')

# Dictionary to map the predicted classes to their labels with emojis
results = {
    0: '✈️ Aeroplane',
    1: '🚗 Automobile',
    2: '🐦 Bird',
    3: '🐱 Cat',
    4: '🦌 Deer',
    5: '🐶 Dog',
    6: '🐸 Frog',
    7: '🐴 Horse',
    8: '🚢 Ship',
    9: '🚚 Truck'
}

# Streamlit app title
st.title('Image Classification App')

# Display the list of classes with emojis
st.subheader('List of Classes that can be Recognized:')
st.markdown(
    """
    - ✈️ **Aeroplane**
    - 🚗 **Automobile**
    - 🐦 **Bird**
    - 🐱 **Cat**
    - 🦌 **Deer**
    - 🐶 **Dog**
    - 🐸 **Frog**
    - 🐴 **Horse**
    - 🚢 **Ship**
    - 🚚 **Truck**
    """,
    unsafe_allow_html=True,
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make a prediction
    pred = model.predict(image_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence_score = pred[0][pred_class] * 100

    # Display the prediction
    st.write(f"Predicted class: {results[pred_class]}")
    st.write(f"Confidence: {confidence_score:.2f}%")

footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
left: 0;
bottom: 0;
width: 100%;
color:white;
text-align: center;
}
</style>
<br><br><br>
<div class="footer">
<p>KRBL-01 Image Classifier <a style='display: block; text-align: center;' href="https://ashimnepal.com.np" target="_blank">Ashim Nepal</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)