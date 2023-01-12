from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import ToTensor
import torch
import os
from torch.cuda.amp import autocast
import importlib
import numpy as np
import argparse
import model.aotgan as net
import cv2



# @st.cache
def load_model(model_name):
    net = importlib.import_module('model.aotgan')
    args = argparse.Namespace()
    args.block_num = 8
    args.rates = [1, 2, 4, 8]
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load('G0000000.pt', map_location='cpu'))
    # half_model = model.half()
    # half_model.eval()
    model.eval()
    return model


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


def infer(img, mask):
    with torch.no_grad():
        img_cv = cv2.resize(np.array(img)[:, :, :3], (512, 512))  # Fixing everything to 512 x 512 for this demo.
        img_tensor = (ToTensor()(img_cv) * 2.0 - 1.0).unsqueeze(0)
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        print(img_tensor.shape)
        print(mask_tensor.shape)

        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))
        comp_np = postprocess(comp_tensor[0])

        return comp_np


stroke_width = 8
stroke_color = "#F00"
bg_color = "#000"
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg", "jpeg"])
sample_bg_image = st.sidebar.selectbox('Sample Images', sorted(os.listdir('./images')))
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 10, 50, 20)


# model_name = st.sidebar.selectbox(
#     "Select model:", ("NimaBoscarino/aot-gan-celebahq", "NimaBoscarino/aot-gan-places2")
# )
model = load_model("G0000000.pt")

bg_image = Image.open(bg_image) if bg_image else Image.open(f"./images/{sample_bg_image}")

st.subheader("Draw on the image to erase features. The inpainted result will be generated and displayed below.")
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=bg_image,
    update_streamlit=True,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    key="canvas",
)

if st.button("Predict"):
# if canvas_result.image_data is not None and bg_image and len(canvas_result.json_data["objects"]) > 0:
    mask = canvas_result.image_data[:, :, 3]
    binary_mask = np.where(mask > 10, 1, 0)

    result = infer(bg_image, binary_mask)

    st.image(result)