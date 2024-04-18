import io
import cv2
import math
import numpy as np
from PIL import Image


def circle_point_px(img, accuracy_angle, r=None):
    rows, cols, _ = img.shape
    assert 360 % accuracy_angle == 0
    x0, y0 = r0, _ = (rows // 2, cols // 2)
    if r:
        r0 = r
    angles = np.arange(0, 360, accuracy_angle)
    cos_angles = np.cos(np.deg2rad(angles))
    sin_angles = np.sin(np.deg2rad(angles))

    x = x0 + r0 * cos_angles
    y = y0 + r0 * sin_angles

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    circle_px_list = img[x, y]
    return circle_px_list


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def HSVDistance(c1, c2):
    y1 = 0.299 * c1[0] + 0.587 * c1[1] + 0.114 * c1[2]
    u1 = -0.14713 * c1[0] - 0.28886 * c1[1] + 0.436 * c1[2]
    v1 = 0.615 * c1[0] - 0.51498 * c1[1] - 0.10001 * c1[2]
    y2 = 0.299 * c2[0] + 0.587 * c2[1] + 0.114 * c2[2]
    u2 = -0.14713 * c2[0] - 0.28886 * c2[1] + 0.436 * c2[2]
    v2 = 0.615 * c2[0] - 0.51498 * c2[1] - 0.10001 * c2[2]
    rlt = math.sqrt((y1 - y2) * (y1 - y2) + (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2))
    return rlt


def crop_to_square(image):
    height, width = image.shape[:2]
    size = min(height, width)
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    cropped = image[start_y:start_y + size, start_x:start_x + size]
    return cropped


def crop_square_from_center(image, size):
    """截取以中心点的正方形"""
    height, width = image.shape[:2]
    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2
    # 计算截取区域的左上角和右下角坐标
    half_size = size // 2
    top_left_x = max(0, center_x - half_size)
    top_left_y = max(0, center_y - half_size)
    bottom_right_x = min(width, center_x + half_size)
    bottom_right_y = min(height, center_y + half_size)
    # 切片截取正方形图像
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return cropped_image


def crop_circle_from_center(image, radius):
    """截取指定半径的圆形"""
    height, width = image.shape[:2]
    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2
    # 创建一个掩码，圆形部分为白色
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    # 使用掩码将原始图像截取为圆形
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # 寻找圆形区域的边界框
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # 裁剪边界框的部分，得到最终的圆形图像
    cropped_circle = masked_image[y:y + h, x:x + w]
    return cropped_circle


def make_circle_transparent(image, radius):
    # 读取图片
    # image = cv2.imread(image_path)

    # 获取图片尺寸
    height, width = image.shape[:2]

    # 分离通道
    b, g, r = cv2.split(image)

    # 创建掩码，将圆形区域设置为白色（255），其余区域为黑色（0）
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, (255), thickness=-1)

    # 创建一个三通道的 alpha 通道图像
    alpha = np.ones((height, width), dtype=np.uint8) * 255
    alpha[mask == 255] = 0

    # 使用 cv2.add 将图像和 alpha 通道相加
    bgra = cv2.merge([b, g, r, alpha])
    result = cv2.add(bgra, np.zeros_like(bgra), mask=alpha)

    return result


def to_cv2_img(image):
    """将输入的图片转换为cv2的格式"""
    if isinstance(image, bytes):
        img_buffer_np = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(img_buffer_np, 1)
    elif isinstance(image, str):
        img = cv2.imread(str(image))
    elif isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img = cv2.imdecode(np.array(bytearray(buf.getvalue()), dtype='uint8'), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f'输入的图片类型无法解析: {type(image)}')
    return img


def discern(inner_image_brg, outer_image_brg, result_img=None, isSingle=True):
    """
    根据背景图片计算单旋转图片的角度。
    """
    # 读取内圈圆形图片
    inner_image_brg = to_cv2_img(inner_image_brg)
    inner_image_brg = crop_circle_from_center(inner_image_brg, 96)  # 将多余白边裁剪
    inner_image_brg = cv2.resize(inner_image_brg, dsize=(200, 200))  # 缩放至与背景图片缺口大小相同
    # 读取背景图片
    outer_image_brg = to_cv2_img(outer_image_brg)
    outer_image_brg = crop_square_from_center(outer_image_brg, 400)  # 截取正方形
    # 图片处理
    inner_image = cv2.cvtColor(inner_image_brg, cv2.COLOR_BGR2HSV)
    outer_image = cv2.cvtColor(outer_image_brg, cv2.COLOR_BGR2HSV)
    all_deviation = []
    total = 360 if isSingle else 180
    for result in range(0, total):
        inner = rotate(inner_image, -result)
        outer = rotate(outer_image, 0 if isSingle else result)
        pic_circle_radius = inner.shape[0] // 2
        inner_circle_point_px = circle_point_px(inner, 1, pic_circle_radius - 5)
        outer_circle_point_px = circle_point_px(outer, 1, pic_circle_radius + 5)
        total_deviation = np.sum(
            [HSVDistance(in_px, out_px) for in_px, out_px in zip(inner_circle_point_px, outer_circle_point_px)])
        all_deviation.append(total_deviation)
    result = all_deviation.index(min(all_deviation))
    if result_img:
        inner = rotate(inner_image_brg, -result)
        outer = rotate(outer_image_brg, 0 if isSingle else result)
        outer = crop_to_square(outer)
        size = inner.shape[0]
        left_point = int((outer.shape[0] - size) / 2)
        right_point = left_point + size
        replace_area = outer[left_point:right_point, left_point:right_point].copy()
        outer[left_point:right_point, left_point:right_point] = replace_area + inner
        cv2.imwrite(result_img, outer)
        cv2.imshow(result_img, outer)
    return result


def discern_one(image, result_img=None, step=1):
    """
    根据背景图片计算单旋转图片的角度。
    """
    # 读取内圈圆形图片
    image = to_cv2_img(image)
    image = cv2.resize(image, (600, 400))
    # cv2.imshow("image", image)
    inner_image_brg = crop_square_from_center(image, 400)  # 截取正方形
    inner_image_brg = crop_circle_from_center(inner_image_brg, 96)  # 将多余白边裁剪
    inner_image_brg = cv2.resize(inner_image_brg, dsize=(200, 200))  # 缩放至与背景图片缺口大小相同
    inner_image_brg = cv2.cvtColor(inner_image_brg, cv2.COLOR_BGR2BGRA)
    # cv2.imshow("inner", inner_image_brg)
    # 读取背景图片
    outer_image_brg = crop_square_from_center(image, 400)  # 截取正方形
    outer_image_brg = make_circle_transparent(outer_image_brg, 100)
    # cv2.imshow("outer", outer_image_brg)

    # 图片处理
    inner_image = cv2.cvtColor(inner_image_brg, cv2.COLOR_BGR2HSV)
    outer_image = cv2.cvtColor(outer_image_brg, cv2.COLOR_BGR2HSV)
    all_deviation = []
    total = 360
    for result in range(0, total, step):
        inner = rotate(inner_image, -result)
        outer = outer_image
        pic_circle_radius = inner.shape[0] // 2
        inner_circle_point_px = circle_point_px(inner, 1, pic_circle_radius - 5)
        outer_circle_point_px = circle_point_px(outer, 1, pic_circle_radius + 5)
        total_deviation = np.sum(
            [HSVDistance(in_px, out_px) for in_px, out_px in zip(inner_circle_point_px, outer_circle_point_px)])
        all_deviation.append(total_deviation)
    result = all_deviation.index(min(all_deviation)) * step
    if result_img:
        inner = rotate(inner_image_brg, -result)
        outer = outer_image_brg
        outer = crop_to_square(outer)
        size = inner.shape[0]
        left_point = int((outer.shape[0] - size) / 2)
        right_point = left_point + size
        replace_area = outer[left_point:right_point, left_point:right_point].copy()
        outer[left_point:right_point, left_point:right_point] = replace_area + inner
        # cv2.imwrite(result_img, outer)
        # cv2.imshow(result_img, outer)
        return result, outer
    return result


def restore_image(inner_image_brg, outer_image_brg, circle_size=100, crop_circle=0, step=1, result_img=None):
    """
    计算旋转图片的角度。
    """
    # 读取内圈圆形图片
    inner_image_brg = to_cv2_img(inner_image_brg)
    circle_size_org = inner_image_brg.shape[0] // 2

    inner_image_brg = crop_circle_from_center(inner_image_brg, circle_size_org - crop_circle)  # 将多余白边裁剪
    inner_image_brg = cv2.resize(inner_image_brg, dsize=(circle_size, circle_size))  # 缩放至与背景图片缺口大小相同

    # 读取背景图片
    outer_image_brg = to_cv2_img(outer_image_brg)
    outer_image_brg = make_circle_transparent(outer_image_brg, circle_size // 2)

    # 图片处理
    inner_image = cv2.cvtColor(inner_image_brg, cv2.COLOR_BGR2HSV)
    outer_image = cv2.cvtColor(outer_image_brg, cv2.COLOR_BGR2HSV)
    all_deviation = []
    total = 360
    for result in range(0, total, step):
        inner = rotate(inner_image, -result)
        outer = outer_image
        pic_circle_radius = inner.shape[0] // 2
        inner_circle_point_px = circle_point_px(inner, 1, pic_circle_radius - 5)
        outer_circle_point_px = circle_point_px(outer, 1, pic_circle_radius + 5)
        total_deviation = np.sum(
            [HSVDistance(in_px, out_px) for in_px, out_px in zip(inner_circle_point_px, outer_circle_point_px)])
        all_deviation.append(total_deviation)
    result = all_deviation.index(min(all_deviation)) * step
    if result_img:
        inner = rotate(inner_image_brg, -result)
        outer = cv2.cvtColor(outer_image_brg, cv2.COLOR_BGRA2BGR)
        outer = crop_to_square(outer)
        size = inner.shape[0]
        left_point = int((outer.shape[0] - size) / 2)
        right_point = left_point + size
        replace_area = outer[left_point:right_point, left_point:right_point].copy()
        outer[left_point:right_point, left_point:right_point] = replace_area + inner
        # cv2.imwrite(result_img, outer)
        # cv2.imshow(result_img, outer)
        return result, outer
    return result


if __name__ == '__main__':
    R = restore_image(
        r'plugins\double_rotate\demo\inner.jpg', 
        r"plugins\double_rotate\demo\bg.jpg", 
        310,
        20,
        6,
        "test"
    )
    print(R)
    cv2.waitKey()
