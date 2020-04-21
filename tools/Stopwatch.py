import time

class Stopwatch:
    t_init = None

    def __init__(self):
        self.t_init = self.start()

    def stop(self, msg=''):
        time_elapsed = time.time() - self.t_start
        print('{} - {}'.format(time_elapsed, msg))
        return time_elapsed
    
    def start(self):
        self.t_start = time.time()
        return self.t_start
    
    def round(self, msg=''):
        time_elapsed = self.stop(msg)
        self.start()
        return time_elapsed
    
    def total(self, msg='Total'):
        time_elapsed = time.time() - self.t_init
        print('{} - {}'.format(time_elapsed, msg))
        return time_elapsed