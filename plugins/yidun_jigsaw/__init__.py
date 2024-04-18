import base64
import io
import json
import os
import time
from io import BytesIO
from PIL import Image
import gradio as gr
from .predict import restore_jigsaw
from utils import logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "网易易盾推理拼图还原"
PLUGIN_VERSION = "1.0.1"
PLUGIN_LABEL = "yidun_jigsaw"


def get_exchangePos(image_data):
    error = 0
    exchangePos = []
    while error <= 10:
        result, out_img = restore_jigsaw(image_data, 80)
        change_num = 0
        for k, v in result.items():
            if k != v:
                change_num += 1
                exchangePos = [v, k]
        if change_num == 2:
            return exchangePos
        error += 1
    return exchangePos


def crop_image(image_path, square_size):
    image = Image.open(image_path)
    width, height = image.size

    # 计算水平和垂直方向上的正方形数量
    num_horizontal_squares = width // square_size
    num_vertical_squares = height // square_size

    # 切割成正方形并保存到列表中
    squares = []
    for i in range(num_vertical_squares):
        for j in range(num_horizontal_squares):
            left = j * square_size
            upper = i * square_size
            right = left + square_size
            lower = upper + square_size
            square = image.crop((left, upper, right, lower))
            squares.append(square)

    return squares


def swap_squares(squares, index1, index2):
    squares[index1], squares[index2] = squares[index2], squares[index1]


def combine_image(squares, num_horizontal_squares, num_vertical_squares, square_size):
    new_image = Image.new('RGB', (num_horizontal_squares * square_size, num_vertical_squares * square_size))

    for i in range(num_vertical_squares):
        for j in range(num_horizontal_squares):
            index = i * num_horizontal_squares + j
            new_image.paste(squares[index], (j * square_size, i * square_size))

    return new_image


def process_and_save_image(input_image, swap):
    square_size = 80
    # 切割图片
    squares = crop_image(io.BytesIO(input_image), square_size)

    # 交换两个正方形的位置
    swap_squares(squares, swap[0], swap[1])

    # 计算水平和垂直方向上的正方形数量
    image = Image.open(io.BytesIO(input_image))
    width, height = image.size
    num_horizontal_squares = width // square_size
    num_vertical_squares = height // square_size

    # 组合图片
    new_image = combine_image(squares, num_horizontal_squares, num_vertical_squares, square_size)

    # 输出图片
    output_buffer = io.BytesIO()
    new_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    return output_buffer.getvalue()


def predict_captcha(img_input: Image.Image):
    t = time.time()
    try:
        if img_input is None:
            raise Exception("Error! 未上传图片!")
        buf = BytesIO()
        img_input.save(buf, format="PNG")
        img_data = buf.getvalue()
        position = get_exchangePos(img_data)
        draw_data = process_and_save_image(img_data, position)
        # 输出PIL格式
        img = Image.open(BytesIO(draw_data))
    except Exception as e:
        logger.error("Error! {}", str(e))
        err = {"error": str(e)}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, json.dumps(position, ensure_ascii=False), f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，版本: {PLUGIN_VERSION}")
    demo_path = os.path.join(CURRENT_PATH, "demo", "0cba1d1b10404a07b95386055d9bd53e.jpg")
    with gr.Row():
        image_input = gr.Image(value=demo_path, sources=["upload"],
                               label="原始图片", type="pil", interactive=True)
        image_output = gr.Image(label="识别结果", type="pil")
    with gr.Row():
        result_output = gr.JSON(label="识别结果")
        result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
    with gr.Row():
        gr.ClearButton(
            [image_input, image_output, result_output, result_time],
            value="清除")
        button = gr.Button("识别测试")
    gr.Markdown(f"[返回主页](/)")
    button.click(predict_captcha, [image_input], [image_output, result_output, result_time])

