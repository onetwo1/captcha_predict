import os
import time
import numpy as np
import gradio as gr
from .predict import restore_image
from utils import logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "双旋转验证码"
PLUGIN_VERSION = "1.1.3"
PLUGIN_LABEL = "double_rotate"


def predict_captcha(inner_image_brg, outer_image_brg, circle_size, crop_circle, step):
    t = time.time()
    try:
        if inner_image_brg is None or outer_image_brg is None:
            raise Exception("Error! 未上传图片!")
        r, img = restore_image(inner_image_brg, outer_image_brg, circle_size, crop_circle, step, True)
    except Exception as e:
        logger.exception("Error! {}", str(e))
        return np.array([]), -1, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, r, f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    with gr.Tab("小红书"):
        gr.Markdown(f"## {PLUGIN_NAME}测试，版本: {PLUGIN_VERSION}")
        inner_path = os.path.join(CURRENT_PATH, "demo", "inner_2.png")
        outer_path = os.path.join(CURRENT_PATH, "demo", "bg_2.png")
        with gr.Row():
            inner_image_input = gr.Image(value=inner_path, sources=["upload"], label="内圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
            outer_image_input = gr.Image(value=outer_path, sources=["upload"], label="外圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
        with gr.Row():
            with gr.Column():
                circle_size_input = gr.Number(label="内圈大小", value=200)
                crop_circle_input = gr.Number(label="裁剪半径(有透明背景或干扰时)", value=105)
                step_input = gr.Number(label="步长(1-10的整数,越低越精准,但耗时越长)", value=6, maximum=10, minimum=1)
            with gr.Column():
                image_output = gr.Image(label="识别结果", type="numpy")
        with gr.Row():
            result_output = gr.Number(label="识别结果", interactive=False)
            result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
        with gr.Row():
            gr.ClearButton([inner_image_input, outer_image_input, image_output, result_output, result_time], value="清除")
            button = gr.Button("识别测试")
        gr.Markdown(f"[返回主页](/)")
        button.click(predict_captcha,
                     [inner_image_input, outer_image_input, circle_size_input, crop_circle_input, step_input],
                     [image_output, result_output, result_time])
    with gr.Tab("顶象"):
        gr.Markdown(f"## {PLUGIN_NAME}测试，版本: {PLUGIN_VERSION}")
        inner_path = os.path.join(CURRENT_PATH, "demo", "inner_3.webp")
        outer_path = os.path.join(CURRENT_PATH, "demo", "bg_3.webp")
        with gr.Row():
            inner_image_input = gr.Image(value=inner_path, sources=["upload"], label="内圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
            outer_image_input = gr.Image(value=outer_path, sources=["upload"], label="外圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
        with gr.Row():
            with gr.Column():
                circle_size_input = gr.Number(label="内圈大小", value=146)
                crop_circle_input = gr.Number(label="裁剪半径(有透明背景或干扰时)", value=12)
                step_input = gr.Number(label="步长(1-10的整数,越低越精准,但耗时越长)", value=6, maximum=10, minimum=1)
            with gr.Column():
                image_output = gr.Image(label="识别结果", type="numpy")
        with gr.Row():
            result_output = gr.Number(label="识别结果", interactive=False)
            result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
        with gr.Row():
            gr.ClearButton([inner_image_input, outer_image_input, image_output, result_output, result_time], value="清除")
            button = gr.Button("识别测试")
        gr.Markdown(f"[返回主页](/)")
        button.click(predict_captcha,
                     [inner_image_input, outer_image_input, circle_size_input, crop_circle_input, step_input],
                     [image_output, result_output, result_time])
    with gr.Tab("超星学习通"):
        gr.Markdown(f"## {PLUGIN_NAME}测试，版本: {PLUGIN_VERSION}")
        inner_path = os.path.join(CURRENT_PATH, "demo", "inner.jpg")
        outer_path = os.path.join(CURRENT_PATH, "demo", "bg.jpg")
        with gr.Row():
            inner_image_input = gr.Image(value=inner_path, sources=["upload"], label="内圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
            outer_image_input = gr.Image(value=outer_path, sources=["upload"], label="外圈图片", type="numpy",
                                         image_mode="RGB", interactive=True)
        with gr.Row():
            with gr.Column():
                circle_size_input = gr.Number(label="内圈大小", value=310)
                crop_circle_input = gr.Number(label="裁剪半径(有透明背景或干扰时)", value=20)
                step_input = gr.Number(label="步长(1-10的整数,越低越精准,但耗时越长)", value=6, maximum=10, minimum=1)
            with gr.Column():
                image_output = gr.Image(label="识别结果", type="numpy")
        with gr.Row():
            result_output = gr.Number(label="识别结果", interactive=False)
            result_time = gr.Textbox(placeholder="识别耗时", label="识别耗时", lines=1, interactive=False)
        with gr.Row():
            gr.ClearButton([inner_image_input, outer_image_input, image_output, result_output, result_time], value="清除")
            button = gr.Button("识别测试")
        gr.Markdown(f"[返回主页](/)")
        button.click(predict_captcha,
                     [inner_image_input, outer_image_input, circle_size_input, crop_circle_input, step_input],
                     [image_output, result_output, result_time])

