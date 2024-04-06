import base64
import json
from io import BytesIO
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template

from plugins.yidun_icon.predict import Siamese, YOLOV5_ONNX
from utils import FONT_PATH, GetResponse, logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "网易易盾图标点选识别"
PLUGIN_VERSION = "1.0.1"
PLUGIN_LABEL = "yidun_icon"

model_path = os.path.join(os.path.dirname(__file__), 'model')
icon_det_path = os.path.join(model_path, "IconDet.onnx")
icon_siamese_path = os.path.join(model_path, "IconSiamese.onnx")
for path in [icon_det_path, icon_siamese_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error! 模型路径无效: '{path}'")
yolo = YOLOV5_ONNX(onnx_path=icon_det_path, classes=['target'])
siamese = Siamese(onnx_path=icon_siamese_path)


def register_predict(app: Flask, executor):
    @app.route('/yidun/icon', methods=['POST'])
    def yidun_icon():
        resp = GetResponse()
        img = request.files.get('image', None)
        if not img:
            resp.data = json.dumps({"code": 500, "message": "未上传图片!"})
            return resp, 500
        try:
            img_data = img.read()
            future = executor.submit(get_icon_position, img_data)
            position, recognition_img = future.result()
            draw_data = draw(recognition_img, position)
            resp.data = json.dumps({
                "code": 200,
                "message": "识别成功!",
                "data": {"position": position, "img": base64.b64encode(draw_data).decode()}
            })
            return resp, 200
        except Exception as e:
            logger.exception(e)
            resp.data = json.dumps({"code": 500, "message": "识别失败!"})
            return resp, 500


def register_page(app: Flask):
    @app.route(f'/{PLUGIN_LABEL}')
    def yidun_icon_page():
        return render_template(f"{PLUGIN_LABEL}.html")


def register_plugin(app: Flask, executor):
    register_predict(app, executor)
    register_page(app)
    logger.success("{}插件加载成功, 版本: v{}", PLUGIN_NAME, PLUGIN_VERSION)


def get_icon_position(bg_img_data):
    """获取图标点选坐标"""
    recognition_img, icon_imgs = split_img(bg_img_data)
    result = yolo.detection(recognition_img)
    img = Image.open(BytesIO(recognition_img))
    siamese_matrix = []
    i = 1
    for j, icon_data in enumerate(icon_imgs):
        icon_img = Image.open(BytesIO(icon_data))
        row = []
        i += 1
        for box in result:
            crop_img = img.crop(box).convert("L")
            s = siamese.reason(icon_img, crop_img)
            row.append(s)
            i += 1
        siamese_matrix.append(row)
    siamese_matrix = np.array(siamese_matrix)
    p = []
    for i in range(siamese_matrix.shape[0]):
        max_index = np.argmax(siamese_matrix[i, :])
        update_matrix(siamese_matrix, (i, max_index))
        p.append(result[max_index])
    points = [[int((p[i][0] + p[i][2]) / 2), int((p[i][1] + p[i][3]) / 2)] for i in range(len(p))]
    return points, recognition_img


def update_matrix(matrix, index):
    """将最大值所在的行和列置为零"""
    matrix[index[0], :] = 0  # 将行置为零
    matrix[:, index[1]] = 0  # 将列置为零
    return matrix


def split_img(img_data, icon_index=0):
    img = Image.open(BytesIO(img_data))
    width, height = img.size
    recognition_area = (0, 0, width, 160)
    icon_area = (0, 160 + icon_index * 20, 75, 160 + (icon_index + 1) * 20)
    recognition_img = img.crop(recognition_area)
    f = BytesIO()
    recognition_img.save(f, format="JPEG")
    recognition_img = f.getvalue()
    icon_img = img.crop(icon_area)
    icon_num = 3
    icon_imgs = []
    for i in range(icon_num):
        icon = icon_img.crop((int(75 / icon_num) * i, 0, int(75 / icon_num) * (i + 1), 20))
        f = BytesIO()
        icon.save(f, format="JPEG")
        icon_imgs.append(f.getvalue())
    return recognition_img, icon_imgs


def draw(img_data, data: dict):
    """绘制识别结果"""
    image = Image.open(BytesIO(img_data)).convert("RGB")
    w, h = image.size
    draw_font = ImageDraw.Draw(image)
    draw_box = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, int(w * 0.04))
    for i, box in enumerate(data):
        char = f"{i}."
        x, y = box
        box_w = int(w * 0.13)
        box_h = int(w * 0.13)
        length = draw_font.textlength(char, font=font)
        text_width, text_height = length, length
        text_bg_x1 = x - 20
        text_bg_y1 = y - 20
        text_bg_x2 = text_bg_x1 + text_width + 4  # 加上一些额外的空间
        text_bg_y2 = text_bg_y1 + text_height + 4  # 加上一些额外的空间
        draw_font.rectangle([(text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2)], fill="black")
        draw_font.text((x - 18, y - 18), char, fill="white", font=font)
        draw_box.rectangle([(x - 20, y - 20), (x + box_w - 20, y + box_h - 20)], outline="blue", width=2)
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer.getvalue()
