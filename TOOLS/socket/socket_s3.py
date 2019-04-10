#-*-coding:utf-8-*-
#并发实例

import socketserver  # 导入socketserver模块


class MyServer(socketserver.BaseRequestHandler):  # 创建一个类，继承自socketserver模块下的BaseRequestHandler类
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
                send_data = input(">>>>>")
                conn.sendall(send_data.encode())
            conn.close()


if __name__ == '__main__':
    sever = socketserver.ThreadingTCPServer(("127.0.0.1", 8880),
                                            MyServer)  # 传入 端口地址 和 我们新建的继承自socketserver模块下的BaseRequestHandler类  实例化对象

    sever.serve_forever()  # 通过调用对象的serve_forever()方法来激活服务端
