import configparser
import os
import sys
from loguru import logger

PROJECT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FONT_PATH =os.path.join(CURRENT_PATH, "res", "SimHei.ttf")
CONFIG_PATH = os.path.join(CURRENT_PATH, "config.ini")


def get_config():
    if os.path.exists(CONFIG_PATH) is False:
        raise FileNotFoundError(f"找不到配置文件! {CONFIG_PATH}")
    # 读取配置文件。
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH, encoding='utf-8')
    return cfg


config = get_config()

# =======日志相关配置=======
SHOW_LOG = True
SAVE_LOG = False
DEBUG = False

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

