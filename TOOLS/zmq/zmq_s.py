import zmq
import cv2
def send_array_and_str(socket, img, string="None", flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(img.dtype),
        shape = img.shape,
    )
    socket.send_string(string, flags | zmq.SNDMORE)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(img, flags, copy=copy, track=track)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:56670")
my_ndarray_image = cv2.imread("d:/downloads/vlcsnap_from_whiteboard.png")

while 1:
    message = socket.recv()
    print(message)
    send_array_and_str(socket, my_ndarray_image)
    #time.sleep(1)
