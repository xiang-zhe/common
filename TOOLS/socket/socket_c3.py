#-*-coding:utf-8-*-

import socket

sk = socket.socket()
sk.connect(("127.0.0.1", 8880))  # 主动初始化与服务器端的连接
while True:
    send_data = input("输入发送内容:")
    #sk.sendall(bytes(send_data)) #python2
    sk.sendall(send_data.encode())
    if send_data == "byebye":
        break
    accept_data = sk.recv(1024)
    print("".join(("接收内容：", str(accept_data.decode()))))
sk.close()
