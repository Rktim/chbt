import streamlit as st
from PIL import Image
import io

st.sidebar.header("Image Processing Options")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif"])


if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.sidebar.checkbox("Reduce Image Size"):
            new_size = st.sidebar.slider("Select new size (in pixels)", 1, 1024, 256)
            image = image.resize((new_size, new_size))
            st.image(image, caption='Reduced Image', use_column_width=True)

       
        if st.sidebar.checkbox("Change Orientation"):
            orientation = st.sidebar.selectbox("Select Orientation", ["Normal", "Rotate 90", "Rotate 180", "Rotate 270"])
            if orientation == "Rotate 90":
                image = image.rotate(90)
            elif orientation == "Rotate 180":
                image = image.rotate(180)
            elif orientation == "Rotate 270":
                image = image.rotate(270)
            st.image(image, caption='Rotated Image', use_column_width=True)

        if st.sidebar.checkbox("Change Format"):
            format_choice = st.sidebar.selectbox("Select Format", ["JPEG", "PNG", "GIF"])
            buf = io.BytesIO()
            if format_choice == "JPEG":
                image.save(buf, format='JPEG')
            elif format_choice == "PNG":
                image.save(buf, format='PNG')
            elif format_choice == "GIF":
                image.save(buf, format='GIF')
            st.download_button("Download Image", buf.getvalue(), f"image.{format_choice.lower()}", f"image/{format_choice.lower()}")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
