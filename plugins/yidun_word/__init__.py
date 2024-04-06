import base64
import json
import time
from io import BytesIO
import os
import random
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from plugins.yidun_word.predict import OCR, YOLOV5_ONNX
from utils import FONT_PATH, logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "网易易盾文字点选识别"
PLUGIN_VERSION = "v1-1-CNN"
PLUGIN_LABEL = "yidun_word"

model_dir = os.path.join(CURRENT_PATH, 'model')
det_path = os.path.join(model_dir, "WordDet.onnx")
if not os.path.exists(det_path):
    raise FileNotFoundError(f"Error! 模型路径无效: '{det_path}'")
imgDet = YOLOV5_ONNX(onnx_path=det_path, classes=["target"])
ocr_model_path = os.path.join(model_dir, 'EfficientNet.onnx')
if not os.path.exists(ocr_model_path):
    raise FileNotFoundError(f"Error! 模型路径无效: '{ocr_model_path}'")
class_name_path = os.path.join(CURRENT_PATH, 'char.json')
if not os.path.exists(class_name_path):
    raise FileNotFoundError(f"Error! 字符集路径无效: '{class_name_path}'")
onnx_init = OCR(ocr_model_path, class_name_path)


def get_position(bg_img_data, fronts):
    """获取点选坐标"""
    positions = imgDet.detection(bg_img_data)
    char_list = []
    img = Image.open(BytesIO(bg_img_data))
    for pos in positions:
        x1, y1, x2, y2 = pos
        crop_img = img.crop((x1, y1, x2, y2))
        crop_img_data = BytesIO()
        crop_img.save(crop_img_data, format="PNG")
        char_list.append({
            'position': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'img_data': crop_img_data.getvalue()
        })
    # 命中的集合
    result = [[] for i in fronts]
    # 未命中集合，当命中的坐标数量不足时，从未命中中选择
    miss_char = []
    for char_data in char_list:
        char = onnx_init.predict(char_data.get('img_data'))
        if char not in fronts:
            miss_char.append(char_data.get('position'))
            continue
        index = fronts.index(char)
        result[index] = char_data.get('position')
    for i, r in enumerate(result):
        if not r:
            result[i] = random.choice(miss_char)
            miss_char.remove(result[i])
    char_result = {}
    for j, pos in enumerate(result):
        char_result[fronts[j]] = pos
    return char_result


def get_word_position(bg_img_data):
    """获取所有文字坐标"""
    positions = imgDet.detection(bg_img_data)
    char_list = []
    img = Image.open(BytesIO(bg_img_data))
    for pos in positions:
        x1, y1, x2, y2 = pos
        crop_img = img.crop((x1, y1, x2, y2))
        crop_img_data = BytesIO()
        crop_img.save(crop_img_data, format="PNG")
        char_list.append({
            'position': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            'img_data': crop_img_data.getvalue()
        })
    result = {}
    for char_data in char_list:
        char = onnx_init.predict(char_data.get('img_data'))
        result[char] = char_data.get('position')
    return result


def draw(img_data, data: dict):
    """绘制识别结果"""
    image = Image.open(BytesIO(img_data)).convert("RGB")
    w, h = image.size
    draw_font = ImageDraw.Draw(image)
    draw_box = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, int(w * 0.04))
    for char, box in data.items():
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


def predict_captcha(img_input: Image.Image, words):
    t = time.time()
    try:
        if img_input is None:
            raise Exception("Error! 未上传图片!")
        buf = BytesIO()
        img_input.save(buf, format="PNG")
        img_data = buf.getvalue()
        if words == '' or words is None:
            position = get_word_position(img_data)
        else:
            position = get_position(img_data, list(words))
        draw_data = draw(img_data, position)
        # 输出PIL格式
        img = Image.open(BytesIO(draw_data))
    except Exception as e:
        logger.error("Error! {}", str(e))
        err = {"error": str(e)}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, json.dumps(position, ensure_ascii=False), f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，模型版本: {PLUGIN_VERSION}")
    prompt_input = gr.Textbox(value="张质身", placeholder="输入提示词", label="提示词", lines=1,
                              interactive=True)
    demo_path = os.path.join(CURRENT_PATH, "demo", "张质身.jpg")
    with gr.Row():
        image_input = gr.Image(value=demo_path, sources=["upload"],
                               label="原始图片", type="pil", interactive=True)
        image_output = gr.Image(label="识别结果", type="pil")
    with gr.Row():
        result_output = gr.JSON(label="识别结果")
        result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
    with gr.Row():
        gr.ClearButton(
            [image_input, prompt_input, image_output, result_output, result_time],
            value="清除")
        button = gr.Button("识别测试")
    button.click(predict_captcha, [image_input, prompt_input], [image_output, result_output, result_time])
