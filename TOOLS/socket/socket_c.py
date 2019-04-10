#-*-coding:utf-8-*-

import socket

sk = socket.socket()
sk.connect(("127.0.0.1", 8886))  # 主动初始化与服务器端的连接
send_data = input("输入发送内容：")
#sk.sendall(bytes(send_data, encoding="utf8"))
#sk.sendall(bytes(send_data)) #python2
sk.sendall(send_data.encode())
accept_data = sk.recv(1024)
print(str(accept_data.decode()))
sk.close()
