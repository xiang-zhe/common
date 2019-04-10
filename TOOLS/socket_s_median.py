#-*-coding:utf-8-*-
#并发实例

import socketserver  # 导入socketserver模块
import socket


def processdata(accept_data):
    # pass
    return accept_data

class MyServer(socketserver.BaseRequestHandler):  # 创建一个类，继承自socketserver模块下的BaseRequestHandler类
    sk = socket.socket()
    sk.connect(("127.0.0.1", 8881))
    def handle(self):  # 要想实现并发效果必须重写父类中的handler方法，在此方法中实现服务端的逻辑代码（不用再写连接准备，包括bind()、listen()、accept()方法）
        while 1:
            conn = self.request
            addr = self.client_address
            # 上面两行代码，等于 conn,addr = socket.accept()，只不过在socketserver模块中已经替我们包装好了，还替我们包装了包括bind()、listen()、accept()方法
            while 1:
                accept_data = str(conn.recv(1024).decode())
                print(accept_data)
                if accept_data == "byebye":
                    break
                else:
                    send_data_sub = processdata(accept_data) 
                    MyServer.sk.sendall(send_data_sub.encode())
                    accept_data_sub = str(MyServer.sk.recv(1024).decode())
                    send_data = processdata(accept_data_sub)
                conn.sendall(send_data.encode())
            conn.close()


if __name__ == '__main__':
    sever = socketserver.ThreadingTCPServer(("127.0.0.1", 8880),
                                            MyServer)  # 传入 端口地址 和 我们新建的继承自socketserver模块下的BaseRequestHandler类  实例化对象

    sever.serve_forever()  # 通过调用对象的serve_forever()方法来激活服务端
