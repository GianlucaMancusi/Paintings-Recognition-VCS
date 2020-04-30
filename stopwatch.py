import time

class Stopwatch:
    t_init = None

    def __init__(self):
        self.t_init = self.start()

    def stop(self):
        time_elapsed = time.time() - self.t_start
        return time_elapsed
    
    def start(self):
        self.t_start = time.time()
        return self.t_start
    
    def round(self):
        time_elapsed = self.stop()
        self.start()
        return time_elapsed
    
    def total(self):
        time_elapsed = time.time() - self.t_init
        return time_elapsed