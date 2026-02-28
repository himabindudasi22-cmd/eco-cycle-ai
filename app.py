import streamlit as st
from ultralytics import YOLO
from PIL import Image
import openai

# Page Setup
st.set_page_config(page_title="EcoCycle AI", page_icon="♻️", layout="centered")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp { background-color: #f9fbf9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("♻️ EcoCycle AI Assistant")
st.subheader("Transform your waste into wonder.")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Camera Input
img_file = st.camera_input("Snap a photo of your item")

if img_file:
    img = Image.open(img_file)
    # Perform detection
    results = model(img)
    
    # Filter detections (Ignore 'person' class, which is index 0 in COCO)
    detected_items = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            # YOLOv8 default 'person' is index 0
            if label != 'person': 
                detected_items.append(label)
    
    unique_items = list(set(detected_items))
    
    if unique_items:
        st.success(f"Detected: {', '.join(unique_items)}")
        
        if st.button("✨ Get Step-by-Step Upcycling Guide"):
            with st.spinner("Brainstorming creative projects..."):
                try:
                    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    prompt = (
                        f"Items detected: {', '.join(unique_items)}. "
                        "Provide 3 creative upcycling projects. For each, include: "
                        "1. Catchy Title. 2. Materials needed. 3. Numbered, detailed step-by-step instructions."
                    )
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.markdown("---")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error("Check your API key/billing quota.")
    else:
        st.warning("No objects detected. Try holding the item closer!")
