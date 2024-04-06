from fastapi import FastAPI
from plugins import yidun_space, yidun_word, yidun_jigsaw
import gradio as gr

app = FastAPI()

app = gr.mount_gradio_app(app, yidun_word.demo, path=f'/{yidun_word.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, yidun_space.demo, path=f'/{yidun_space.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, yidun_jigsaw.demo, path=f'/{yidun_jigsaw.PLUGIN_LABEL}')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=1012,
    )
