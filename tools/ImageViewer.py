import matplotlib.pyplot as plt
import math

class ImageViewer:
    counter = 0

    def __init__(self, img_num, cols=4):
        self.img_num = img_num
        self.cols = cols
        self.f, self.axarr = plt.subplots(math.ceil(img_num / cols), cols)
    
    def add(self, img, title=''):
        y = self.counter % self.cols
        x = self.counter // self.cols
        self.counter += 1
        if self.img_num <= self.cols:
            self.axarr[y].imshow(img)
            self.axarr[y].set_title(title)
        else:
            self.axarr[x, y].imshow(img)
            self.axarr[x, y].set_title(title)
    
    def show(self):
        plt.show()