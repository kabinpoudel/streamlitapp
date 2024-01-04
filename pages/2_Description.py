import streamlit as st


st.set_page_config(
    page_title="Nepali OCR",
    page_icon="🇳🇵",
    layout="wide",
)

st.sidebar.markdown('# Description')

html_str = f"""
<p>
<div style="text-align: justify">
The dataset used for this project comprised of 11,600 grayscale images of handwritten nepali characters. The characters are consonants, numerals and vowels.
</div>

* consonants = "क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह क्ष त्र ज्ञ"
* numerals = "० १ २ ३ ४ ५ ६ ७ ८ ९"
* vowels = "अ आ इ ई उ ऊ ए ऐ ओ औ अं अः"

<div style="text-align: justify">
The dataset comprised of 58 classes( 36 consonants + 10 numerals + 12 vowels). Each class consists of 200 images splitted into training set comprising of 160 images and test set comprising of 40 images.
</div>

The model used for classification was [TinyVGG](https://github.com/poloclub/cnn-explainer/blob/master/tiny-vgg/tiny-vgg.py). This model can be used to perform classification on the aforementioned 58 classes with 83.30% accuracy.

</p>
"""
st.markdown(html_str, unsafe_allow_html=True)
