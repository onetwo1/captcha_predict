import hashlib
from io import BytesIO
import json
import os
import time
import gradio as gr
from PIL import Image
from .predict import NineClassify
from utils import logger

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "极验4九宫格识别"
PLUGIN_VERSION = "v2_fp16"
PLUGIN_LABEL = "geetest4_nine"


nine_model_path = os.path.join(CURRENT_PATH, "model", "geetest4_nine_v2_fp16.onnx")
ncls = NineClassify(nine_model_path)


def get_nine_position(que_data, imgs, nine_nums):
    positions = []
    p = {
        0: [1, 1],
        1: [1, 2],
        2: [1, 3],
        3: [2, 1],
        4: [2, 2],
        5: [2, 3],
        6: [3, 1],
        7: [3, 2],
        8: [3, 3],
    }
    que_class = ncls.predict(que_data)["class"]
    result = ncls.predict_list(imgs)
    for item in result:
        if item.get("class") == que_class:
            index = item.get("index")
            positions.append(p[index])
    c = nine_nums - len(positions)
    if c > 0:
        result.sort(key=lambda x: x.get("confidence"))
        for item in result:
            if item.get("class") == que_class:
                continue
            positions.append(p[item.get("index")])
            if len(positions) >= nine_nums:
                break
    elif c < 0:
        positions = positions[:c]
    return positions, que_class


def get_images(positions, imgs):
    result = []
    for pos in positions:
        index = (pos[0] - 1) * 3 + (pos[1] - 1)
        result.append(Image.open(BytesIO(imgs[index])))
    return result


def update_matrix(matrix, index):
    """将最大值所在的行和列置为零"""
    matrix[index[0], :] = 0  # 将行置为零
    matrix[:, index[1]] = 0  # 将列置为零
    return matrix


def split_image(image_data):
    # 打开图像
    img = Image.open(BytesIO(image_data))
    # 获取图像的宽度和高度
    width, height = img.size
    # 计算每个小块的宽度和高度
    block_width = width // 3
    block_height = height // 3
    # 由于保存切割后的图片
    split_image_list = []

    # 切割图像为9份
    for i in range(3):
        for j in range(3):
            # 计算切割区域的坐标
            left = j * block_width
            top = i * block_height
            right = (j + 1) * block_width
            bottom = (i + 1) * block_height

            # 切割图像
            block = img.crop((left, top, right, bottom))
            # 调整图片大小
            block = block.resize((128, 128))
            data = BytesIO()
            block.save(data, format='PNG')
            split_image_list.append(data.getvalue())

    return split_image_list


def predict_captcha(img_input: Image.Image, icon_input: Image.Image, nine_nums: int):
    t = time.time()
    try:
        if img_input is None:
            raise Exception("Error! 未上传图片!")
        if 9 < nine_nums and nine_nums <= 0:
            raise Exception("Error! nine_nums范围错误!")
        buf = BytesIO()
        img_input.save(buf, format="PNG")
        imgs = split_image(buf.getvalue())
        positions, que_class = get_nine_position(icon_input, imgs, nine_nums)
        imgs_output = get_images(positions, imgs)
    except Exception as e:
        logger.exception(e)
        err = {"error": str(e)}
        return img_input, err, "", f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return imgs_output, json.dumps(positions, ensure_ascii=False), que_class, f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，模型版本: {PLUGIN_VERSION}")
    demo_path_0 = os.path.join(CURRENT_PATH, "demo", "0072b074e4b0491fb7bcd91a4af7a748.jpg")
    demo_path_1 = os.path.join(CURRENT_PATH, "demo", "698777432d4b6352e008a1d267329aa1.png")
    with gr.Row():
        icon_input = gr.Image(
            value=demo_path_1, 
            sources=["upload"], label="目标图片", type="pil", image_mode="RGBA", interactive=True)
        image_input = gr.Image(
            value=demo_path_0, 
            sources=["upload"], label="原始图片", type="pil", image_mode="RGBA", interactive=True)
    nine_nums = gr.Number(value=3, label="目标数量(默认为3)", interactive=True)
    with gr.Row():
        image_output = gr.Gallery(label="识别结果")
        with gr.Column():
            result_output = gr.JSON(label="识别结果")
            result_class = gr.Textbox(placeholder="", label="识别类型", lines=1, interactive=False)
            result_time = gr.Textbox(placeholder="", label="识别耗时", lines=1, interactive=False)
    with gr.Row():
        gr.ClearButton(
            [image_input, icon_input, image_output, result_output, result_class, result_time],
            value="清除")
        button = gr.Button("识别测试")
    gr.Markdown(f"[返回主页](/)")
    button.click(predict_captcha, [image_input, icon_input, nine_nums], [image_output, result_output, result_class, result_time])



