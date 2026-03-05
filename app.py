import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# --- 1. SETUP ---
# Replace 'YOUR_API_KEY' with your actual Gemini API Key
genai.configure(api_key="YOUR_API_KEY")
model_gemini = genai.GenerativeModel('gemini-1.5-flash')
# Load the YOLO model (the "brain" for seeing objects)
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

model = load_yolo()

st.title("EcoCycle AI ♻️")

# --- 2. CAMERA INPUT ---
picture = st.camera_input("Take a picture of your item")

if picture:
    # Convert picture to a format YOLO understands
    img = Image.open(picture)

    # --- 3. DETECTION CODE ---
    results = model.predict(source=img)
    
    found_objects = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            found_objects.append(label)
    
    unique_objects = list(set(found_objects)) # Removes duplicates

    # --- 4. SELECTION WIDGET ---
    if unique_objects:
        st.write("### I found these items!")
        
        user_selection = st.multiselect(
            "Which one do you want to upcycle?", 
            options=unique_objects
        )

        if user_selection:
            st.success(f"Selected: {', '.join(user_selection)}")
            
            # --- 5. SEND TO GEMINI ---
            # This button triggers the AI instructions
            if st.button("✨ Get Step-by-Step Upcycling Guide"):
                items_string = ", ".join(user_selection)
                
                prompt = f"""
                I have the following items: {items_string}. 
                Please provide 3 creative, eco-friendly upcycling projects. 
                For each project, include:
                - A catchy title
                - A list of extra materials needed
                - Clear step-by-step instructions
                """
                
                with st.spinner("Thinking of creative ideas..."):
                    try:
                        response = model_gemini.generate_content(prompt)
                        st.markdown("---")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")
    else:
        st.warning("I couldn't find any objects. Try moving the camera closer!")
