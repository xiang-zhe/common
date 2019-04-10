#-*-coding:utf-8-*-


import socket

sk = socket.socket()
sk.bind(("127.0.0.1", 9008))
sk.listen(5)
while True:

    conn, addr = sk.accept()
    print('waiting for connecting...')
    while True:
        accept_data = str(conn.recv(1024).decode())
        print("".join(["接收内容：", accept_data, "     客户端口：", str(addr[1])]))
        if accept_data == "byebye":  # 如果接收到“byebye”则跳出循环结束和第一个客户端的通讯，开始与下一个客户端进行通讯
            break
        send_data = input("输入发送内p容：")
        #conn.sendall(bytes(send_data)) #python2 bytes
        conn.sendall(send_data.encode())
    conn.close()  # 跳出循环时结束通讯
