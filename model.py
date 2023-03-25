# 以下を「model.py」に書き込み

#TensorFlow,TensorFlow Hub
import tensorflow as tf
import tensorflow_hub as hub

# イメージの出力のためのライブラリ
import matplotlib.pyplot as plt
# イメージのダウンロードのためのライブラリ
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
# イメージの出力のためのライブラリ
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def display_image(image):

    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def download_and_resize_image(
    img, new_width=256, new_height=256, display=False):
#    '''イメージをダウンロードして指定のサイズにリサイズする
#    '''
    _, filename = tempfile.mkstemp(suffix=".jpg")
    image_data =img
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    return pil_image  # イメージのパスを返す


def draw_bounding_box_on_image(image,
                               ymin, xmin, ymax, xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
#    '''イメージ上にバウンディングボックスを描画する
#    '''
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top),(left, top)],
        width=thickness,
        fill=color)

    # バウンディングボックスの上部に表示するラベルが画像の上部を超える場合は、
    # ラベルをバウンディングボックスの下部に表示する
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # ラベルに表示する文字列の上下にマージンを設定
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # ラベルに表示する文字列のリストを逆順で出力
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                    fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=5, min_score=0.1):
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf',
            25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                        int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

def detectors():
# ssd+mobilenet V2: 小規模で高速
# https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1

#  FasterRCNN+InceptionResNet V2を使用
#    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 
    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    detector = hub.load(module_handle).signatures['default']
    return detector



