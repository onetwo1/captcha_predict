import base64
import json
import time
from io import BytesIO
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from .predict import ColorClassify, YOLOV5_ONNX
from utils import FONT_PATH, logger
import logging
import jieba

jieba.setLogLevel(logging.INFO)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "网易易盾空间推理识别"
PLUGIN_VERSION = "v1-1"
PLUGIN_LABEL = "yidun_space"

model_dir = os.path.join(CURRENT_PATH, 'model')
space_det_path = os.path.join(model_dir, "SpaceDet.onnx")
color_model_path = os.path.join(model_dir, "ColorClassify.onnx")
for path in [space_det_path, color_model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error! 模型路径无效: '{path}'")
space_obj_classes = ["A_侧向", "A_正向", "B_侧向", "B_正向", "C_侧向", "C_正向", "D_侧向", "D_正向", "E_侧向", "E_正向",
                     "F_侧向", "F_正向", "G_侧向", "G_正向", "H_侧向", "H_正向", "I_侧向", "I_正向", "J_侧向", "J_正向",
                     "K_侧向", "K_正向", "L_侧向", "L_正向", "M_侧向", "M_正向", "N_侧向", "N_正向", "O_侧向", "O_正向",
                     "P_侧向", "P_正向", "Q_侧向", "Q_正向", "R_侧向", "R_正向", "S_侧向", "S_正向", "T_侧向", "T_正向",
                     "U_侧向", "U_正向", "V_侧向", "V_正向", "W_侧向", "W_正向", "X_侧向", "X_正向", "Y_侧向", "Y_正向",
                     "Z_侧向", "Z_正向", "a_侧向", "a_正向", "b_侧向", "b_正向", "c_侧向", "c_正向", "d_侧向", "d_正向",
                     "e_侧向", "e_正向", "f_侧向", "f_正向", "g_侧向", "g_正向", "h_侧向", "h_正向", "i_侧向", "i_正向",
                     "j_侧向", "j_正向", "k_侧向", "k_正向", "l_侧向", "l_正向", "m_侧向", "m_正向", "n_侧向", "n_正向",
                     "o_侧向", "o_正向", "p_侧向", "p_正向", "q_侧向", "q_正向", "r_侧向", "r_正向", "s_侧向", "s_正向",
                     "t_侧向", "t_正向", "u_侧向", "u_正向", "v_侧向", "v_正向", "w_侧向", "w_正向", "x_侧向", "x_正向",
                     "y_侧向", "y_正向", "z_侧向", "z_正向", "0_侧向", "0_正向", "1_侧向", "1_正向", "2_侧向", "2_正向",
                     "3_侧向", "3_正向", "4_侧向", "4_正向", "5_侧向", "5_正向", "6_侧向", "6_正向", "7_侧向", "7_正向",
                     "8_侧向", "8_正向", "9_侧向", "9_正向", "圆柱_侧向", "圆柱_正向", "圆锥_侧向", "圆锥_正向",
                     "球_侧向", "球_正向", "立方体_侧向", "立方体_正向"]
space_yolo = YOLOV5_ONNX(onnx_path=space_det_path, classes=space_obj_classes)
cc = ColorClassify(color_model_path)


class CaptchaException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def space_detection(imgae):
    """获取空间推理图片中的目标位置"""
    return space_yolo.decect(imgae)


def color_classify(image):
    """颜色分类"""
    return cc.predict(image)


def get_space_position(prompt: str, image: bytes):
    sp = SpacePredict(prompt, image)
    return sp.predict()


def get_all_position(image: bytes):
    obj_list = predict_obj(image)
    new_obj = {}
    for i, item in enumerate(obj_list):
        pos = item.get("position")
        new_obj[f"{i}."] = [int((pos[0] + pos[2]) / 2), int((pos[1] + pos[3]) / 2)]
    return new_obj


def predict_obj(image: bytes):
    output = space_detection(image)
    if not output:
        raise CaptchaException("Error! 未识别到物体")
    obj_list = [{"obj": item.get("classes"), "position": item.get("crop")} for item in output]
    color_name = {"red": "红色", "blue": "蓝色", "yellow": "黄色", "green": "绿色", "gray": "灰色"}
    new_obj_list = []
    img = cv2.imdecode(np.array(bytearray(image), dtype='uint8'), cv2.IMREAD_COLOR)
    for obj_ in obj_list:
        x1, y1, x2, y2 = obj_.get("position")
        crop_img = img[y1:y2, x1:x2]
        color = color_name[color_classify(crop_img)]
        obj = obj_.get("obj")
        new_obj = obj_.copy()
        new_obj["obj"] = obj + "_" + color
        new_obj["words"] = new_obj["obj"].split("_")
        new_obj_list.append(new_obj)
    # draw(image, obj_list)
    # draw(image, new_obj_list)
    return new_obj_list


class SpacePredict(object):
    def __init__(self, prompt: str, image: bytes):
        self.prompt = prompt
        self.image = image
        self.obj_list = predict_obj(image)

        # 形状相同的物体（易识别错的）
        self.same_objs = [
            [
                ['o', '小写'],
                ['O', '大写'],
                ['0', '数字']
            ],
            [
                ['x', '小写'],
                ['X', '大写'],
            ],
            [
                ['w', '小写'],
                ['W', '大写'],
            ],
            [
                ['l', '小写'],
                ['I', '大写'],
            ],
            [
                ['v', '小写'],
                ['V', '大写'],
            ],
            [
                ['z', '小写'],
                ['Z', '大写'],
            ],
            [
                ['s', '小写'],
                ['S', '大写'],
            ],
            [
                ['k', '小写'],
                ['K', '大写'],
            ],
            [
                ['c', '小写'],
                ['C', '大写'],
            ]
        ]

    def predict(self):
        target_obj = self.predict_prompt()
        x1, y1, x2, y2 = target_obj.get("position")
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return [x, y]

    def predict_prompt(self):
        """通过类型推理图片"""
        if "颜色一样的" in self.prompt:
            prompt = self.prompt.replace("请点击", "")
            reference, target = prompt.split("颜色一样的")
            return self.reference(reference, target, "color")
        elif "朝向一样的" in self.prompt:
            prompt = self.prompt.replace("请点击", "")
            reference, target = prompt.split("朝向一样的")
            return self.reference(reference, target, "direction")
        else:
            target = self.prompt.replace("请点击", "")
            obj_list = self.feature_search(target)
            target_obj = self.search_in_objs_no_reference(target, obj_list)
            if target_obj is None:
                raise CaptchaException("未能找到目标物体!")
            return target_obj

    def reference(self, reference, target, ptype):
        new_reference_list = self.feature_search(reference)
        new_target_list = self.feature_search(target)
        new_reference = new_reference_list[0]
        target_obj = self.search_in_objs(new_reference, new_target_list, ptype)
        if target_obj is None:
            raise CaptchaException("未能找到目标物体!")
        return target_obj

    def feature_search(self, new_prompt):
        target_obj, target_feature, target_feature_type = self.get_obj(new_prompt)
        if target_feature_type == "dirction":
            new_feature = self.search_in_image(target_obj, target_direction=target_feature)
        elif target_feature_type == "color":
            new_feature = self.search_in_image(target_obj, target_color=target_feature)
        else:
            new_feature = self.search_in_image(target_obj)
        return new_feature

    def get_obj(self, new_prompt):
        """通过特征和物体提示词获取物体数据"""
        words = list(jieba.cut(new_prompt))
        if "的" in words:
            words.remove("的")
        target_obj = words[-1]
        if len(words) == 1:
            # 仅有参照物
            return [target_obj, None, None]
        if len(words) == 2:
            # 大小写字母 or 数字
            if target_obj not in ["立方体", "圆柱", "圆锥", "球"]:
                return [target_obj, None, None]
            # 特征 + 三维物体
            feature = words[0]
            feature_type = "direction" if "向" in feature else "color"
            return [target_obj, feature, feature_type]
        if len(words) == 3:
            # 特征 + 大小写字母 or 数字
            feature = words[0]
            feature_type = "direction" if "向" in feature else "color"
            return [target_obj, feature, feature_type]
        else:
            raise CaptchaException(f"未知的目标提示词: {new_prompt}")

    def search_in_image(self, target_obj_name, target_direction=None, target_color=None, same_mode=True):
        result = []
        for item in self.obj_list:
            words = item.get("words")
            obj_name = words[0]
            direction = words[1]
            color = words[2]
            if obj_name == target_obj_name and target_direction is None and target_color is None:
                result.append(item)
                # break
            elif obj_name == target_obj_name and target_direction is not None and target_color is None:
                if target_direction == direction:
                    result.append(item)
                    # break
            elif obj_name == target_obj_name and target_direction is None and target_color is not None:
                if target_color == color:
                    result.append(item)
                    # break
            else:
                pass
        if len(result) > 0 or not same_mode:
            return result
        same_obj = None
        for sames in self.same_objs:
            for same in sames:
                if target_obj_name == same[0]:
                    sames.remove(same)
                    same_obj = sames
                    break
        if same_obj is None:
            raise CaptchaException("图片中没有符号条件的物体!")
        for new_target_obj_name, otype in same_obj:
            # 将图片中的new_target_obj_name替换为target_obj_name
            new_obj_list = []
            for item in self.obj_list:
                if new_target_obj_name not in item.get("words"):
                    continue
                words = item.get("words")
                words[0] = target_obj_name
                item["words"] = words
                item["obj"] = "_".join(words)
                new_obj_list.append(item)
            self.obj_list = new_obj_list
            new_result = self.search_in_image(target_obj_name, target_direction, target_color, same_mode=False)
            if new_result:
                return new_result
        raise CaptchaException("图片中没有符号条件的物体!")

    def search_in_objs(self, reference, targets, ptype):
        if ptype == "direction":
            reference_direction = reference.get("words")[1]
            for target in targets:
                target_direction = target.get("words")[1]
                if reference_direction == target_direction:
                    return target
        elif ptype == "color":
            reference_color = reference.get("words")[2]
            for target in targets:
                target_color = target.get("words")[2]
                if reference_color == target_color:
                    return target
        else:
            raise CaptchaException(f"未知参照物类型: {ptype}")

    def search_in_objs_no_reference(self, target_prompt, obj_list):
        target_obj, target_feature, target_feature_type = self.get_obj(target_prompt)
        if target_feature_type == "dirction":
            for target in obj_list:
                if target_feature == target.get("words")[1]:
                    return target
        elif target_feature_type == "color":
            for target in obj_list:
                if target_feature == target.get("words")[2]:
                    return target
        else:
            for target in obj_list:
                if target_obj == target.get("words")[0]:
                    return target


def draw(img_data, data):
    """绘制识别结果"""
    image = Image.open(BytesIO(img_data)).convert("RGB")
    w, h = image.size
    draw_font = ImageDraw.Draw(image)
    draw_box = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, int(w * 0.04))
    for i, box in enumerate(data):
        char = "目标"
        x, y = box
        box_w = int(w * 0.14)
        box_h = int(w * 0.14)
        length = draw_font.textlength(char, font=font)
        text_width, text_height = length, int(w * 0.04)
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
            raise CaptchaException("Error! 未上传图片!")
        buf = BytesIO()
        img_input.save(buf, format="PNG")
        img_data = buf.getvalue()
        if words == '' or words is None:
            raise CaptchaException("未输入提示词!")
        position = get_space_position(words, img_data)
        draw_data = draw(img_data, [position])
        # 输出PIL格式
        img = Image.open(BytesIO(draw_data))
    except CaptchaException as e:
        logger.error("Error! {}", str(e))
        err = {"error": str(e)}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    except Exception as e:
        logger.error("Error! {}", str(e))
        err = {"error": "发生错误!"}
        return img_input, err, f"耗时: {(time.time() - t) * 1000:.2f}ms"
    return img, json.dumps(position, ensure_ascii=False), f"耗时: {(time.time() - t) * 1000:.2f}ms"


with gr.Blocks(title=f"验证码识别测试-{PLUGIN_NAME}") as demo:
    gr.Markdown(f"## {PLUGIN_NAME}测试，模型版本: {PLUGIN_VERSION}")
    prompt_input = gr.Textbox(value="请点击数字2颜色一样的小写z", placeholder="输入提示词", label="提示词", lines=1,
                              interactive=True)
    demo_path = os.path.join(CURRENT_PATH, "demo", "请点击数字2颜色一样的小写z.jpg")
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


