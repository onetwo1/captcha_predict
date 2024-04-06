import configparser
import json
import os
import sys
from flask import Flask
from concurrent.futures import ThreadPoolExecutor
from utils import config, logger, GetResponse
from flask_limiter import Limiter, RequestLimit
from flask_limiter.util import get_remote_address

try:
    IP = config.get('setting', 'ip')
    PORT = config.getint('setting', 'port')
    max_workers = config.getint('setting', 'max_workers')
    enable_limit = config.getboolean('limit', 'enable')
    day_limit = config.get('limit', 'day_limit')
    hour_limit = config.get('limit', 'hour_limit')
except Exception as e:
    logger.error(e)
    logger.error("读取配置文件错误! 请检查配置文件路径或内容是否正确。")
    sys.exit(1)

executor = ThreadPoolExecutor(max_workers=max_workers)  # 可以根据需要调整线程数量
app = Flask("API")

if enable_limit:
    # IP限制
    def default_error_responder(request_limit: RequestLimit):
        resp = GetResponse()
        resp.data = json.dumps({"code": 429, "message": "请求频率过高"})
        resp.status_code = 429
        return resp
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[f"{day_limit} per day", f"{hour_limit} per hour"],
        on_breach=default_error_responder
    )
    logger.success("IP限制启动成功! 每天限制次数:{}, 每小时限制次数:{}", day_limit, hour_limit)

logger.info("正在加载插件...")
# 注册路由
from plugins import yidun_word, yidun_icon, yidun_jigsaw, yidun_space

yidun_word.register_plugin(app, executor)
yidun_icon.register_plugin(app, executor)
yidun_jigsaw.register_plugin(app, executor)
yidun_space.register_plugin(app, executor)

logger.success("全部插件加载完成!")

if __name__ == '__main__':
    logger.success("API已启动 ==> http://{}:{}", IP, PORT)
    app.run(host=IP, port=PORT, debug=False)
