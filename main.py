import streamlit as st
import torch
import torchvision
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srgan_checkpoint = "./models/checkpoint_srgan.pth_25.tar"
srresnet_checkpoint = "./models/checkpoint_srresnet_129.pth.tar"
#load models
srgan = torch.load(srgan_checkpoint,map_location=torch.device(device))['generator']
srgan.eval()
srresnet = torch.load(srresnet_checkpoint,map_location=torch.device(device))['model']

IMAGENAME = 'img.jpg'
FONT_SIZE = 100

st.set_page_config(
    page_title="Nepali OCR",
    page_icon="🇳🇵",
    layout="wide",
)


st.title("Nepali Character Recognition")

st.sidebar.markdown('# Nepali OCR')

uploaded_file = st.file_uploader(label="Upload your image file",
                 type=['jpg', 'jpeg'],
                 accept_multiple_files=False,
                 key="uploaded_file",
                 on_change=None,
                 args=None,
                 kwargs=None,
                 disabled=False,
                 label_visibility="visible"
)


col1,col2,col3 = st.columns(3)

with col1:
    st.write("Image")
    if uploaded_file is not None:
        with open(IMAGENAME,'wb+') as f:
            f.write(uploaded_file.read())
        
    st.image(IMAGENAME, caption="Input Image")
    
def perform_superresolvesrgan():
    hr_img = Image.open(IMAGENAME, mode="r")
    hr_img = hr_img.convert('RGB')
    sr_img_srgan = srgan(convert_image(hr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    return sr_img_srgan

def perform_superresolvesrresnet():
    hr_img = Image.open(IMAGENAME, mode="r")
    hr_img = hr_img.convert('RGB')
    sr_img_srresnet = srresnet(convert_image(hr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
    return sr_img_srresnet
    # st.image(sr_img_srresnet, caption="Input Image")
with col2:
    st.write("Select Model")
    choice=st.selectbox(" ",("srresnet","srgan"))   
    convertbtn=st.button(label="Convert to High Quality", key="btn")
with col3:
    st.write("SuperResolved Image")
    if convertbtn:
        if choice == "srresnet":
            superimage=perform_superresolvesrresnet()
            st.image(superimage, caption="Output Image")
        else:
            superimage=perform_superresolvesrgan()
            st.image(superimage, caption="Output Image")




