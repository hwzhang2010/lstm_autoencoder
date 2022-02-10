import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options
import json

from tornado.options import options
from datetime import datetime

import detect

tornado.options.define("port", default=8000, type=int, help="端口")

class IndexHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, DELETE, PUT')
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")

    def prepare(self):
        if self.request.headers.get("Content-Type").startswith("application/json"):
            self.json_dict = json.loads(self.request.body)
        else:
            self.json_dict = None


class ElecHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, DELETE, PUT')
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")

    def prepare(self):
        if self.request.headers.get("Content-Type").startswith("application/x-www-form-urlencoded"):
            self.json_dict = json.loads(self.request.body)
        else:
            self.json_dict = None

    def post(self):
        if self.json_dict:
            for key, value in self.json_dict.items():
                print(key, value)

        starttime = datetime.now()
        flag = detect.detect_elec()
        endtime = datetime.now()
        print("电源分系统检测耗时(ms): %d" % ((endtime - starttime).seconds * 1000 + (endtime - starttime).microseconds / 1000))

        print(flag)
        self.finish({'result': flag})


class TempHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, DELETE, PUT')
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")

    def prepare(self):
        if self.request.headers.get("Content-Type").startswith("application/x-www-form-urlencoded"):
            self.json_dict = json.loads(self.request.body)
        else:
            self.json_dict = None

    def post(self):
        if self.json_dict:
            for key, value in self.json_dict.items():
                print(key, value)

        starttime = datetime.now()
        flag = detect.detect_temp()
        endtime = datetime.now()
        print("热控分系统检测耗时(ms): %d" % ((endtime - starttime).seconds * 1000 + (endtime - starttime).microseconds / 1000))

        print(flag)
        self.finish({'result': flag})

class AdcsHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, DELETE, PUT')
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")

    def prepare(self):
        if self.request.headers.get("Content-Type").startswith("application/x-www-form-urlencoded"):
            self.json_dict = json.loads(self.request.body)
        else:
            self.json_dict = None

    def post(self):
        if self.json_dict:
            for key, value in self.json_dict.items():
                print(key, value)

        starttime = datetime.now()
        flag = detect.detect_adcs()
        endtime = datetime.now()
        print("姿控分系统检测耗时(ms): %d" % ((endtime - starttime).seconds * 1000 + (endtime - starttime).microseconds / 1000))

        print(flag)
        self.finish({'result': flag})



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tornado.options.parse_command_line()
    # 创建一个应用对象
    app = tornado.web.Application([(r'/', IndexHandler), (r"/api/elec", ElecHandler), (r"/api/temp", TempHandler), (r"/api/adcs", AdcsHandler), ])
    # 创建httpserver实例
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.instance().start()


