from threading import Thread

def thread_decorator(func):
    def wrap_func(*args, **kwargs):
        Thread(target = func, args = args, kwargs = kwargs).start()
    return wrap_func
