# 验证码识别测试
> 本项目旨在提供验证码识别测试，包括易盾、极验、其他验证码等

![Author](https://img.shields.io/badge/Author-z5882852-blue)
![Version](https://img.shields.io/github/v/release/z5882852/captcha_predict?label=Version)
![GitHub License](https://img.shields.io/github/license/z5882852/captcha_predict)


![GitHub Repo stars](https://img.shields.io/github/stars/z5882852/captcha_predict)
![GitHub forks](https://img.shields.io/github/forks/z5882852/captcha_predict)
![GitHub watchers](https://img.shields.io/github/watchers/z5882852/captcha_predict)



## 如何使用


1. **文件拉取**：
    
    - ~~通过`git clone`命令，请确保安装了 Git LFS~~（Git LFS流量有限，如果不成功就尝试其他方式）
    - [百度云盘](https://pan.baidu.com/s/18EwTDqHW5vkILL77iSSr_A) 提取码：811f
    - [阿里云盘](https://www.alipan.com/s/uyHsdrMJeYw) 提取码：j7l2 （exe自解压文件，仅支持windows）

2. **安装环境**

    1. 安装Python3 (版本建议3.10以上)
    2. 安装依赖
        ```bash
        pip install -r requirements.txt
        ```

3. **运行项目**

    可以通过多种运行
    - 使用py文件运行
        ```bash
        python app.py
        ```
      
    - 使用`uvicorn`运行，监听公网建议使用该命令
        ```bash
        uvicorn app:app --host 0.0.0.0 --port 1012
        ```

## 插件
插件目录结构
```
项目
└─plugins
   └─plugin_name  # 插件名，可以自定义
       ├─__init__.py  # 插件入口文件
       ├─*.py  # 其他的py文件
       ├─demo  # 存放演示图片
       └─model  # 存放模型文件
```
插件入口文件
必须包含以下变量
```
PLUGIN_NAME     插件名
PLUGIN_VERSION  插件版本
PLUGIN_LABEL    插件标签(与web路径相同)
demo            gradio页面(参考其他插件)
```

例如
```python
PLUGIN_NAME = "极验4九宫格识别"
PLUGIN_VERSION = "v2_fp16"
PLUGIN_LABEL = "geetest4_nine"

# ...
# 其他内容

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
```

当添加新的插件以后，请在`index.py`添加插件路径和封面
> 未来可能重构成自动添加的

## 其他

本人精力有限，未来可能不会再维护项目。

欢迎大佬们提交PR
