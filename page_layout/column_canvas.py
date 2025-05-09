import streamlit as st
from streamlit_drawable_canvas import st_canvas

def column_canvas(): 
    st.header("Draw Here")
    st.write("(Sketch of an apple or a pear)") 

    canvas_result = st_canvas(
        fill_color="white",  # Set the background color
        stroke_width=13,  # Set the stroke width
        stroke_color="black",  # Set the stroke color
        background_color="white",  # Background color
        width=192,  # Width of the canvas
        height=192,  # Height of the canvas
        drawing_mode="freedraw",  # Drawing mode
        key="canvas",  # Key to access the canvas state
    )
    return canvas_result

def column_canvas_adversarial(): 
    st.header("Draw Here")
    st.latex(r"x")
    st.markdown("<br>", unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="white",  # Set the background color
        stroke_width=13,  # Set the stroke width
        stroke_color="black",  # Set the stroke color
        background_color="white",  # Background color
        width=192,  # Width of the canvas
        height=192,  # Height of the canvas
        drawing_mode="freedraw",  # Drawing mode
        key="canvas",  # Key to access the canvas state
    )


    return canvas_result

