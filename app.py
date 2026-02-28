import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# 1. Setup API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 3. Widget (This creates 'img_file')
img_file = st.camera_input("Snap a photo of your item")

# 4. Logic (Only run if img_file exists)
if img_file:
    img = Image.open(img_file)
    results = model(img)
    
    # Filter detections (Ignore person)
    detected_items = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if label != 'person': 
                detected_items.append(label)
    
    unique_items = list(set(detected_items))
    
    if unique_items:
        st.success(f"Detected: {', '.join(unique_items)}")
        
        if st.button("✨ Get Step-by-Step Upcycling Guide"):
            with st.spinner("Brainstorming with Gemini..."):
                try:
                    model_gen = genai.GenerativeModel('gemini-1.5-flash-latest')
                    prompt = (f"Items: {', '.join(unique_items)}. Provide 3 DIY projects. "
                              "Include: 1. Title, 2. Materials, 3. Numbered steps.")
                    response = model_gen.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")
