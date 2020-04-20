import time

class Stopwatch:
    def __init__(self):
        self.start()

    def stop(self, msg=''):
        print('{} - {}'.format(time.time() - self.t0, msg))
    
    def start(self):
        self.t0 = time.time()
    
    def round(self, msg=''):
        self.stop(msg)
        self.start()