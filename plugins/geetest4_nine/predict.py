import io
import os
import cv2
import onnxruntime
import numpy as np
from PIL import Image


class NineClassify(object):
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error! 模型路径无效: '{model_path}'")
        self._session = onnxruntime.InferenceSession(model_path)
        self.size = 128
        self.classify = {0: '七星瓢虫', 1: '乌龟', 2: '书', 3: '井盖', 4: '企鹅', 5: '伞', 6: '信号灯', 7: '兔子', 8: '公交车', 9: '公鸡', 10: '冰箱', 11: '剪刀', 12: '叉子', 13: '口红', 14: '台灯', 15: '台球', 16: '听诊器', 17: '地球仪', 18: '头盔', 19: '帽子', 20: '手套', 21: '手电筒', 22: '手表', 23: '打印机', 24: '拉链', 25: '插排', 26: '摩托车', 27: '救护车', 28: '斧头', 29: '方向盘', 30: '旅行箱', 31: '望远镜', 32: '桌子', 33: '桥', 34: '梳子', 35: '椅子', 36: '气球', 37: '水池', 38: '注射器', 39: '火箭', 40: '烟斗', 41: '热水壶', 42: '照相机', 43: '熊猫', 44: '牙刷', 45: '牛', 46: '狗', 47: '狮子', 48: '猪', 49: '猫', 50: '猴子', 51: '电钻', 52: '眼镜', 53: '碗', 54: '秋千', 55: '积木', 56: '笔', 57: '纽扣', 58: '羊', 59: '羽毛球', 60: '老虎', 61: '船', 62: '苍蝇拍', 63: '蝴蝶', 64: '螺丝刀', 65: '袋鼠', 66: '袜子', 67: '计算器', 68: '订书机', 69: '贝壳', 70: '足球', 71: '轮椅', 72: '轮胎', 73: '过山车', 74: '钥匙', 75: '钱包', 76: '铁轨', 77: '锅', 78: '锅铲', 79: '键盘', 80: '长颈鹿', 81: '音响', 82: '领带', 83: '骆驼', 84: '鱼', 85: '鳄鱼', 86: '鸟', 87: '鹿', 88: '鼠标', 89: '齿轮'}
        self.fp16 = False
        self.check_fp16()
    
    def check_fp16(self):
        # 检查是否为fp16
        tensor_type = self._session.get_inputs()[0].type
        if tensor_type == 'tensor(float16)':
            self.fp16 = True
    
    def transparence2white(self, img: Image.Image):
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
        
    def read_img(self, image):
        """
        转换图片格式
        """
        if isinstance(image, bytes):
            return self.read_img(Image.open(io.BytesIO(image)))
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            image = self.transparence2white(image)
            image.save(buf, format="PNG")
            img = cv2.imdecode(np.array(bytearray(buf.getvalue()), dtype='uint8'), cv2.IMREAD_COLOR)
        elif isinstance(image, str):
            return self.read_img(Image.open(image))
        else:
            raise ValueError(f"Error! 图片格式错误: {type(image)}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size)).astype(np.float16 if self.fp16 else np.float32) / 255
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img

    def predict_all(self, image):
        img = self.read_img(image)
        ouput = self._session.run(None, {"images": img})[0][0]
        data = {}
        for i, confidence in enumerate(ouput):
            data[self.classify[i]] = float(confidence)
        return data

    def predict(self, image):
        img = self.read_img(image)
        ouput = self._session.run(None, {"images": img})[0][0]
        index = np.argmax(ouput)
        data = {
            "confidence": ouput[index],  # 置信度
            "class": self.classify[index],  # 类型
        }
        return data

    def predict_list(self, image_list):
        output = []
        for i, image in enumerate(image_list):
            result = self.predict(image)
            result["index"] = i
            output.append(result)
        return output

