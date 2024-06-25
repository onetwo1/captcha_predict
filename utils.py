from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
import gradio as gr
import sys
from loguru import logger
import importlib
import time
import os


PROJECT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FONT_PATH =os.path.join(CURRENT_PATH, "res", "SimHei.ttf")


# =======日志相关配置=======
SHOW_LOG = True
SAVE_LOG = False
DEBUG = True

logger.remove(0)
level = "DEBUG" if DEBUG else "INFO"
if SHOW_LOG:
    console_log_handler = logger.add(sys.stderr, level=level, enqueue=False)
if SAVE_LOG:
    LOG_DIR = os.path.join(PROJECT_DIR, "logs")
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_PATH = os.path.join(LOG_DIR, "log_{time}.log")
    file_log_handler = logger.add(LOG_PATH, level=level, encoding="utf-8", enqueue=False)

logger.debug("日志初始化成功!")


# 动态加载plugins目录下的所有插件
def load_plugins(directory, fastapi_app):
    logger.info(f"需要加载 {len(os.listdir(directory))} 个插件")
    with Progress(
            "[progress.description]{task.description}({task.completed}/{task.total})",
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[bright_green]{task.fields[info]}"),
    ) as progress:
        task = progress.add_task("[cyan]加载插件中", total=len(os.listdir(directory)), info="-")
        for module_name in os.listdir(directory):
            try:
                st = time.time()
                module_path = f"{directory}.{module_name}"
                module = importlib.import_module(module_path)
                if hasattr(module, 'demo') and hasattr(module, 'PLUGIN_LABEL'):
                    fastapi_app = gr.mount_gradio_app(fastapi_app, module.demo, path=f'/{module.PLUGIN_LABEL}')
                    progress.console.print(
                        f"[bright_green]插件 {module_name:<15} 加载成功! 用时: {int(1000 * (time.time() - st))}ms")
                    progress.update(task, info=module.PLUGIN_NAME)
                else:
                    logger.error(f"插件 {module_name:<15} 加载失败! 插件缺失demo和PLUGIN_LABEL")
            except ImportError as e:
                logger.error(f"插件加载失败! ImportError: {e}")
            progress.update(task, advance=1)
        progress.update(task, info="", description="[green]加载完成")
    logger.success("所有插件加载完成!")
    return fastapi_app
