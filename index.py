import os
import gradio as gr

img_dir = os.path.join("res", "images")

with gr.Blocks(title=f"验证码识别测试") as index:
    gr.Markdown(f"""# 验证码识别测试""")
    gr.Markdown(f"""> 本项目旨在提供验证码识别测试，包括易盾、极验、其他验证码等""")

    with gr.Accordion("网易易盾", open=True):
        with gr.Row():
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "yidun_word.jpg"), show_label=False, show_download_button=False)
                gr.Button("易盾文字点选测试", size="lg", link="/yidun_word")
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "yidun_icon.jpg"), show_label=False, show_download_button=False)
                gr.Button("易盾图标点选测试", size="lg", link="/yidun_icon")
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "yidun_space.jpg"), show_label=False, show_download_button=False)
                gr.Button("易盾空间推理测试", size="lg", link="/yidun_space")
        with gr.Row():
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "yidun_jigsaw.jpg"), show_label=False, show_download_button=False)
                gr.Button("易盾推理拼图测试", size="lg", link="/yidun_jigsaw")
            with gr.Column():
                pass
            with gr.Column():
                pass
    with gr.Accordion("极验4代", open=False):
        with gr.Row():
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "geetest4_word.jpg"), show_label=False, show_download_button=False)
                gr.Button("极验4文字点选测试", size="lg", link="/geetest4_word")
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "geetest4_icon.jpg"), show_label=False, show_download_button=False)
                gr.Button("极验4图标点选识别测试", size="lg", link="/geetest4_icon")
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "geetest4_nine.jpg"), show_label=False, show_download_button=False)
                gr.Button("极验4九宫格识别测试", size="lg", link="/geetest4_nine")

    with gr.Accordion("其他验证码", open=False):
        with gr.Row():
            with gr.Column():
                gr.Image(value=os.path.join(img_dir, "double_rotate.png"), show_label=False, show_download_button=False)
                gr.Button("双旋转验证码测试", size="lg", link="/double_rotate")
            with gr.Column():
                pass
            with gr.Column():
                pass


