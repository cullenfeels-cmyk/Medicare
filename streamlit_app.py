import streamlit as st
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Medicare AI Portal", page_icon="🏥", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Sidebar Navigation across all templates
st.sidebar.title("Medicare AI Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["Home", "About", "Contact", "Authentication / Login", "User Dashboard", "Admin Dashboard"]
)

# Map radio selection to template filenames
page_map = {
    "Home": "index.html",
    "About": "about.html",
    "Contact": "contact.html",
    "Authentication / Login": "auth.html",
    "User Dashboard": "dashboard.html",
    "Admin Dashboard": "admin.html"
}

target_file = page_map.get(page, "index.html")
html_path = os.path.join(TEMPLATE_DIR, target_file)

# Load and render the selected HTML template
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Render full HTML template layout
    components.html(html_content, height=850, scrolling=True)
else:
    st.error(f"Template file '{target_file}' not found in the 'templates/' folder.")