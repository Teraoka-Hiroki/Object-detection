# 以下を「app.py」に書き込み
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
#import model
from model import draw_boxes
from model import detectors
from model import draw_bounding_box_on_image
#from model import download_and_resize_image
from model import display_image

#5. 物体検出に使用する画像データを検出用にデコードする関数

def load_img(img):
    img = np.asarray(img).astype(np.float32)
    return img



#6. 物体検出を実行し、バウンディングボックスが描画された画像を出力する関数
def run_detector(detector,img):
    img =img
    img = load_img(img)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}

    # バウンディングボックスを画像上に描画する
    image_with_boxes = draw_boxes(

        img, result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])
    # バウンディングボックスが描画された画像を出力

    return image_with_boxes


st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("物体検出アプリ")
st.sidebar.write("「モデル：ssd+mobilenet V2（小規模で高速）」を使用")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))

#detect = detectors()

if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["jpg"])
elif img_source == "カメラで撮影":
#    device = user_input = st.text_input("input your video/camera device", "0")
#    if device.isnumeric():
#        device = int(device)
#    cap = cv2.VideoCapture(device)
#    img_file = cap

    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測

#7. 物体検出を実行し、検出結果を反映させた画像を出力

        detect = detectors()
        image = img.resize((640, 480))
        result_img = run_detector(detect,image)

        image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('temporary.jpg', image)
        # 結果の表示
        st.subheader("判定結果")
        st.image('temporary.jpg', caption="検出の画像", width=480)
