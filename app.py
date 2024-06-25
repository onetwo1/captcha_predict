from index import index
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
from utils import load_plugins, gr

app = FastAPI()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/index/")


app = gr.mount_gradio_app(app, index, path='/index/')

# 加载plugins目录下的所有插件
app = load_plugins("plugins", app)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=1012,
        # reload=True,
    )
