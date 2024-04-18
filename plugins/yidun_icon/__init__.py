import base64
import json
from io import BytesIO
import os
import time
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .predict import Siamese, YOLOV5_ONNX
from utils import FONT_PATH, logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "网易易盾图标点选识别"
PLUGIN_VERSION = "v1-2_fp16"
PLUGIN_LABEL = "yidun_icon"

model_path = os.path.join(os.path.dirname(__file__), 'model')
icon_det_path = os.path.join(model_path, "yidun_icon_det_v1.onnx")
icon_siamese_path = os.path.join(model_path, "yidun_icon_siamese_v2_fp16.onnx")
for path in [icon_det_path, icon_siamese_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error! 模型路径无效: '{path}'")
yolo = YOLOV5_ONNX(onnx_path=icon_det_path, classes=['target'])
siamese = Siamese(onnx_path=icon_siamese_path)


def get_icon_position(bg_img_data):
    """获取图标点选坐标"""
    recognition_img, icon_imgs = split_img(bg_img_data)
    result = yolo.detection(recognition_img)
    img = Image.open(BytesIO(recognition_img))
    siamese_matrix = []
    for j, icon_data in enumerate(icon_imgs):
        crop_img_list = []
        for box in result:
            crop_img = img.crop(box).convert("L")
            crop_img_list.append(crop_img)
        row = siamese.reason_all(icon_data, crop_img_list)
        siamese_matrix.append(row)
    siamese_matrix = np.array(siamese_matrix)
    p = []
    for i in range(siamese_matrix.shape[0]):
        max_index = np.argmax(siamese_matrix[i, :])
        update_matrix(siamese_matrix, (i, max_index))
        p.append(result[max_index])
    points = [[int((p[i][0] + p[i][2]) / 2), int((p[i][1] + p[i][3]) / 2)] for i in range(len(p))]
    return points, recognition_img


def get_icon_position_fast(bg_img_data):
    """获取图标点选坐标(快速模式)"""
    recognition_img, icon_imgs = split_img(bg_img_data)
    result = yolo.detection(recognition_img)
    img = Image.open(BytesIO(recognition_img))
    crop_img_list = []
    for box in result:
        crop_img = img.crop(box).convert("L")
        crop_img_list.append(crop_img)
    excluded_indices = []  # 存储已经排除的列索引
    for j, icon_data in enumerate(icon_imgs):
        row = siamese.reason_all(icon_data, crop_img_list)
        # 找出每行最大的元素所在的列
        max_index = np.argmax(row)
        # 将已经排除的列设为0.0
        for index in excluded_indices:
            row.insert(index, 0.0)
        # 排除已选择的列
        del crop_img_list[max_index]
        # 将该列索引加入排除列表
        excluded_indices.append(np.argmax(row))
    points = [[int((result[i][0] + result[i][2]) / 2), int((result[i][1] + result[i][3]) / 2)] for i in excluded_indices]
    
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


def predict_captcha(img_input: Image.Image, fast_mode):
    t = time.time()
    try:
        if img_input is None:
            raise Exception("Error! 未上传图片!")
        buf = BytesIO()
        img_input.save(buf, format="PNG")
        img_data = buf.getvalue()
        if fast_mode:
            position, recognition_img = get_icon_position_fast(img_data)
        else:
            position, recognition_img = get_icon_position(img_data)
        draw_data = draw(recognition_img, position)
        # 输出PIL格式
        img = Image.open(BytesIO(draw_data))
    except Exception as e:
        logger.exception(e)
        err = {"error": str(e)}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, json.dumps(position, ensure_ascii=False), f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，模型版本: {PLUGIN_VERSION}")
    demo_path = os.path.join(CURRENT_PATH, "demo", "9c7e375dced6860e5bfc105633ea711f.jpg")
    with gr.Row():
        image_input = gr.Image(value=demo_path, sources=["upload"],
                               label="原始图片", type="pil", interactive=True)
        image_output = gr.Image(label="识别结果", type="pil")
    fast_mode = gr.Checkbox(label="快速模式，启用快速模式后会稍微增加识别速度，但是精度会降低。")
    with gr.Row():
        result_output = gr.JSON(label="识别结果")
        result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
    with gr.Row():
        gr.ClearButton(
            [image_input, image_output, result_output, result_time],
            value="清除")
        button = gr.Button("识别测试")
    gr.Markdown(f"[返回主页](/)")
    button.click(predict_captcha, [image_input, fast_mode], [image_output, result_output, result_time])

