import streamlit as st
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai  # Add this

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ... [Keep your YOLO loading and camera input code here] ...

if img_file:
    # ... [Keep your object detection logic] ...
    
    if unique_items:
        st.success(f"Detected: {', '.join(unique_items)}")
        
        # New Gemini Logic
        if st.button("✨ Generate Step-by-Step Upcycling Guide"):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"The camera detected: {', '.join(unique_items)}. "
                    "Act as an expert Upcycling Assistant. Provide 3 detailed DIY projects. "
                    "For each project, include: 1. Catchy Title, 2. Materials needed, 3. Numbered, detailed step-by-step instructions. Keep it encouraging!"
                )
                
                with st.spinner("Brainstorming with Gemini..."):
                    response = model.generate_content(prompt)
                    st.markdown("---")
                    st.subheader("🛠️ Your Upcycling Roadmap")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"Error: {e}")
