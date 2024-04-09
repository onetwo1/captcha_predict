from fastapi import FastAPI
from index import index
from plugins import yidun_space, yidun_word, yidun_jigsaw, yidun_icon
from plugins import geetest4_word, geetest4_nine, geetest4_icon
import gradio as gr

app = FastAPI()


app = gr.mount_gradio_app(app, yidun_word.demo, path=f'/{yidun_word.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, yidun_space.demo, path=f'/{yidun_space.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, yidun_jigsaw.demo, path=f'/{yidun_jigsaw.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, yidun_icon.demo, path=f'/{yidun_icon.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, geetest4_nine.demo, path=f'/{geetest4_nine.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, geetest4_word.demo, path=f'/{geetest4_word.PLUGIN_LABEL}')
app = gr.mount_gradio_app(app, geetest4_icon.demo, path=f'/{geetest4_icon.PLUGIN_LABEL}')

app = gr.mount_gradio_app(app, index, path='/index.html')

    
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=1012,
    )
