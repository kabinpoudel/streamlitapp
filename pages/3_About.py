import streamlit as st

st.set_page_config(
    page_title="Nepali OCR",
    page_icon="🇳🇵",
    layout="wide",
)

st.markdown("## About")

col1, col2 = st.columns(2)

with col1:
    html_str = f"""
    <div >
    <div style="font-family: 'Oswald', sans-serif; font-size: 32px;"><b>Pukar Karki</b></div>
    <div style="font-family: 'Oswald', sans-serif; font-size: 20px;"><b>Assistant Professor</b></div>
    <p><b>pukar@ioepc.com</b><br>
    <p>
    Department of Electronics and Computer Engineering<br>
    Purwanchal Campus, IOE, Tribhuvan University<br>
    56700 Gangalal Marga Tinkune<br>
    Dharan-8, Sunsari, Nepal<br>
    </p>
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)

with col2:
    st.markdown("### Connect with me.")
    html_str_2 = f"""
    <div class="col-md-4" style="margin-top:2%">
        <dd><a href="https://scholar.google.com/citations?user=Fy8bz8YAAAAJ&hl=en">Google Scholar</a></dd> 
        <dd><a href="https://www.researchgate.net/profile/Pukar-Karki">Research Gate</a></dd> 
        <dd><a href="https://github.com/pukarkarki/">Github</a></dd> 
        <dd><a href="https://np.linkedin.com/in/pukarkarki">LinkedIn</a></dd>
    </div>
    """
    st.markdown(html_str_2, unsafe_allow_html=True)


st.sidebar.markdown('# About')

