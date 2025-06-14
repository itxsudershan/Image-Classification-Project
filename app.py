import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

# App configuration
st.set_page_config(page_title="Veggie/Fruit Classifier", layout="centered")
st.title('🍎🥦 Image Classification Model – Veggies & Fruits')

def load_model_safely():
    """Try multiple possible model locations"""
    possible_paths = [
        'Image_classify.keras',              # Current directory
        './Image_classify.keras',            # Current directory (explicit)
        'IP PROJECT/Image_classify.keras',   # Project subfolder
        'Image_Class.Model.h5',              # Possible alternative format
        'Image_classify.h5'                  # Another possible format
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return load_model(path), None
            except Exception as e:
                return None, f"Found model but couldn't load: {str(e)}"
    
    return None, "Model not found in any standard locations"

# Load model with better error handling
model, error = load_model_safely()

if model is None:
    st.error(f"❌ Model loading failed:\n{error}")
    st.markdown("""
    ### Troubleshooting Steps:
    1. Make sure `Image_classify.keras` is in the same folder as this script
    2. Check if you need to retrain the model using `Image_Class.Model.ipynb`
    3. Verify the file isn't corrupted
    """)
    st.stop()

# Class labels (updated with emojis for better visualization)
data_cat = [
    '🍎 apple', '🍌 banana', '🔴 beetroot', '🫑 bell pepper', '🥬 cabbage', '🫑 capsicum',
    '🥕 carrot', '🥦 cauliflower', '🌶️ chilli pepper', '🌽 corn', '🥒 cucumber', '🍆 eggplant',
    '🧄 garlic', '🫚 ginger', '🍇 grapes', '🌶️ jalepeno', '🥝 kiwi', '🍋 lemon', '🥬 lettuce',
    '🥭 mango', '🧅 onion', '🍊 orange', '🌶️ paprika', '🍐 pear', '🫛 peas', '🍍 pineapple',
    '🍈 pomegranate', '🥔 potato', '🍐 raddish', '🫘 soy beans', '🥬 spinach', '🌽 sweetcorn',
    '🍠 sweetpotato', '🍅 tomato', '🥔 turnip', '🍉 watermelon'
]

# Image configuration
img_height = 180
img_width = 180

# File uploader with drag-and-drop support
uploaded_file = st.file_uploader(
    "📤 Upload an image of a vegetable or fruit",
    type=['jpg', 'jpeg', 'png']
)

# Also keep the text input option for local testing
image_path = st.text_input(
    '📝 Or enter local file path (for testing)',
    'apple.jpg'
)

def predict_image(img_path):
    """Process and predict an image"""
    try:
        # Load and preprocess image
        img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_batch = tf.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch)
        score = tf.nn.softmax(predictions[0])
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", width=250)
        with col2:
            st.success(f"**Prediction:** {data_cat[np.argmax(score)]}")
            st.write(f"**Confidence:** {np.max(score) * 100:.2f}%")
            
            # Show top 3 predictions
            top_indices = np.argsort(score)[::-1][:3]
            st.write("Top Predictions:")
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. {data_cat[idx]} ({score[idx] * 100:.1f}%)")
                
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Handle either uploaded file or local path
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    predict_image(temp_path)
    os.remove(temp_path)  # Clean up
elif os.path.exists(image_path):
    predict_image(image_path)
else:
    st.warning("⚠️ Please upload an image or provide a valid file path")