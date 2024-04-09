import json
import os
import time
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from plugins.geetest4_word.predict import Predict
from utils import FONT_PATH, logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "极验4文字点选识别"
PLUGIN_VERSION = "v1-3"
PLUGIN_LABEL = "geetest4_word"


word_path = os.path.join(CURRENT_PATH, "models", "geetest4_word_det_v1.onnx")
siamese_path = os.path.join(CURRENT_PATH, "models", "geetest4_word_siamese_v3.onnx")
for path in [siamese_path, word_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error! 模型路径无效: '{path}'")
pre_word = Predict(per_path=siamese_path, yolo_path=word_path)


def get_word_position(captcha_img_data, icon_img_list: list[Image.Image]):
    que_img_list = [transparence2white(icon) for icon in icon_img_list]
    result = pre_word.run2(captcha_img_data, que_img_list)
    # positions = []
    # for point in result:
    #     center_x = (point[0] + point[2]) / 2
    #     center_y = (point[1] + point[3]) / 2
    #     positions.append([int(center_x), int(center_y)])
    return result



def transparence2white(img):
    sp = img.size
    width = sp[0]
    height = sp[1]
    for yh in range(height):
        for xw in range(width):
            dot = (xw, yh)
            color_d = img.getpixel(dot)
            if color_d[3] == 0:
                color_d = (255, 255, 255, 255)
                img.putpixel(dot, color_d)
    return img


def draw(image, data: list):
    """绘制识别结果"""
    w, h = image.size
    draw_font = ImageDraw.Draw(image)
    draw_box = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, int(w * 0.04))
    for i, box in enumerate(data):
        char = f"{i}."
        x1, y1, x2, y2 = box
        length = draw_font.textlength(char, font=font)
        text_bg_x2 = x1 + length + 4  # 加上一些额外的空间
        text_bg_y2 = y1 + length + 4  # 加上一些额外的空间
        draw_font.rectangle([(x1, y1), (text_bg_x2, text_bg_y2)], fill="black")
        draw_font.text((x1 + 4, y1 + 2), char, fill="white", font=font)
        draw_box.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
    return image


def predict_captcha(img_input: Image.Image, icon_input_1, icon_input_2, icon_input_3):
    t = time.time()
    try:
        if None in [img_input, icon_input_1, icon_input_2, icon_input_3]:
            raise Exception("Error! 未上传图片!")
        positions = get_word_position(img_input, [icon_input_1, icon_input_2, icon_input_3])
        img = draw(img_input, positions)
    except Exception as e:
        logger.exception(e)
        err = {"error": str(e)}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, json.dumps(positions, ensure_ascii=False), f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，模型版本: {PLUGIN_VERSION}")
    demo_path = os.path.join(CURRENT_PATH, "demo", "1.jpg")
    icon_path_1 = os.path.join(CURRENT_PATH, "demo", "1_1.png")
    icon_path_2 = os.path.join(CURRENT_PATH, "demo", "1_2.png")
    icon_path_3 = os.path.join(CURRENT_PATH, "demo", "1_3.png")
    with gr.Row():
        icon_input_1 = gr.Image(
            value=icon_path_1, 
            sources=["upload"], label="目标文字-1", type="pil", image_mode="RGBA", interactive=True)
        icon_input_2 = gr.Image(
            value=icon_path_2, 
            sources=["upload"], label="目标文字-2", type="pil", image_mode="RGBA", interactive=True)
        icon_input_3 = gr.Image(
            value=icon_path_3, 
            sources=["upload"], label="目标文字-3", type="pil", image_mode="RGBA", interactive=True)
    with gr.Row():
        image_input = gr.Image(
            value=demo_path, 
            sources=["upload"], label="原始图片", type="pil", image_mode="RGBA", interactive=True)
        image_output = gr.Image(label="识别结果", type="pil")
    with gr.Row():
        result_output = gr.JSON(label="识别结果")
        result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
    with gr.Row():
        gr.ClearButton(
            [image_input, icon_input_1, icon_input_2, icon_input_3, image_output, result_output, result_time],
            value="清除")
        button = gr.Button("识别测试")
    gr.Markdown(f"[返回主页](/)")
    button.click(predict_captcha, [image_input, icon_input_1, icon_input_2, icon_input_3], [image_output, result_output, result_time])



