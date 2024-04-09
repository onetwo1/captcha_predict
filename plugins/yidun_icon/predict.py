import io
import json
import os
import time

import cv2
import onnxruntime
import numpy as np
from PIL import Image


def rescale_boxes(boxes, current_dim, original_shape):
    """Rescales bounding boxes to the original shape"""
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def tag_images(imgs, img_detections, img_size, classes, max_prob=0.5):
    imgs = [imgs]

    """图片展示"""
    results = []
    zero = lambda x: int(x) if x > 0 else 0
    if img_detections is None:
        return results

    for img_i, (img, detections) in enumerate(zip(imgs, img_detections)):
        # Create plot
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_pred in detections:
                if conf > max_prob:
                    results.append(
                        {
                            "crop": [zero(i) for i in (x1, y1, x2, y2)],
                            "classes": classes[int(cls_pred)],
                            "prob": conf,
                        }
                    )
        else:
            print("识别失败")
    return results


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:  # 还有数据
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
        left_index = (np.where(IOU <= thresh))[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]

    return np.array(keep)



def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1)
            j = np.expand_dims(j, 0).T
            x = np.concatenate((box, conf, j), 1)[conf.reshape(1, -1)[0] > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded
        return output


class YOLOV5_ONNX(object):
    def __init__(self, onnx_path, classes, providers=None):
        """初始化onnx"""
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        self.classes = classes
        self.img_size = 320

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_tensor):
        """获取输入tensor"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_tensor
        return input_feed

    def letterbox(
        self,
        img,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        """图片归一化"""
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return img, ratio, (dw, dh)

    def to_numpy(self, img, shape):
        # 超参数设置
        img_size = shape
        src_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 图片填充并归一化
        img = self.letterbox(src_img, img_size, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def decect(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(io.BytesIO(file))
        else:
            img = Image.open(file)
        img = img.convert("RGB")
        img = np.array(img)
        image_numpy = self.to_numpy(img, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(image_numpy)
        pred = self.onnx_session.run(None, input_feed)[0]
        pred = non_max_suppression(pred, 0.5, 0.5)
        res = tag_images(img, pred, self.img_size, self.classes, 0.5)
        return res

    def detection(self, file):
        output = self.decect(file)
        position = [item.get('crop') for item in output]
        return position


class Siamese(object):
    def __init__(self, onnx_path, providers=None):
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_shape = [105, 105]
        self.fp16 = False
        self.check_fp16()

    def check_fp16(self):
        # 检查是否为fp16
        tensor_type = self.sess.get_inputs()[0].type
        if tensor_type == 'tensor(float16)':
            self.fp16 = True

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def open_pillow(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(io.BytesIO(file))
        elif isinstance(file, Image.Image):
            img = file
        else:
            img = Image.open(file)
        return img

    def open_image(self, file, input_shape, nc=3):
        out = self.open_pillow(file)
        # 改变大小 并保证其不失真
        out = out.convert("RGB")
        h, w = input_shape
        out = out.resize((w, h), 1)
        if nc == 1:
            out = out.convert("L")
        return out

    def set_img(self, lines):
        image = self.open_image(lines, self.input_shape, 3)
        if self.fp16:
            image = np.array(image).astype(np.float16) / 255.0
        else:
            image = np.array(image).astype(np.float32) / 255.0
        photo = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        return photo

    def reason(self, image_1, image_2):
        photo_1 = self.set_img(image_1)
        photo_2 = self.set_img(image_2)
        out = self.sess.run(None, {"x1": photo_1, "x2": photo_2})
        out = out[0]
        out = self.sigmoid(out)
        out = out[0][0]
        return out

    def reason_all(self, image_1, image_2_list):
        photo_1 = self.set_img(image_1)
        photo_2_all = None
        photo_1_all = photo_1
        for image_2 in image_2_list:
            photo_2 = self.set_img(image_2)
            if photo_2_all is None:
                photo_2_all = photo_2
            else:
                photo_2_all = np.concatenate((photo_2_all, photo_2))
                photo_1_all = np.concatenate((photo_1_all, photo_1))
        out = self.sess.run(None, {"x1": photo_1_all, "x2": photo_2_all})
        out = out[0]
        out = self.sigmoid(out)
        out = out.tolist()
        out = [i[0] for i in out]
        return out



