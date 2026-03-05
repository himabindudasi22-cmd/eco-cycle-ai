import streamlit as st
from ultralytics import YOLO
from PIL import Image  # <--- THIS IS THE MISSING LINE
import google.generativeai as genai

# Your model loading code follows...
import streamlit as st
from ultralytics import YOLO
# ... other imports ...

# --- 1. Load your Model ---
model = YOLO('yolov8n.pt') 

# --- 2. The Camera/Upload Section ---
picture = st.camera_input("Take a picture of your item")

if picture:
    # Convert picture to a format YOLO understands
    img = Image.open(picture)

    # --- 3. PASTE THE DETECTION CODE HERE ---
    results = model.predict(source=img)
    
    found_objects = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            found_objects.append(label)
    
    unique_objects = list(set(found_objects)) # Removes duplicates

    # --- 4. PASTE THE SELECTION WIDGET HERE ---
    if unique_objects:
        st.write("I found these items!")
        
        # This creates the box for you to click
        user_selection = st.multiselect(
            "Which one do you want to upcycle?", 
            options=unique_objects
        )

        if user_selection:
            st.success(f"Selected: {', '.join(user_selection)}")
            
            # --- 5. SEND TO GEMINI ---
            # Now you can use 'user_selection' in your prompt to Gemini
            # Example: "How can I upcycle a " + str(user_selection) + "?"
    else:
        st.warning("I couldn't find any objects. Try moving the camera closer!")
