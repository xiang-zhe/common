import zmq
import numpy as np

def recv_array_and_str(socket, flags=0, copy=True, track=False):
    string = socket.recv_string(flags=flags)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    img = np.frombuffer(bytes(memoryview(msg)), dtype=md['dtype'])
    return string, img.reshape(md['shape'])

context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REQ)
#socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://localhost:56670")
while True:
    socket.send("input1".encode("utf-8"))
    string, img = recv_array_and_str(socket)
    print("Received reply: ",img)  ##.decode('utf-8')
    #time.sleep(1)
