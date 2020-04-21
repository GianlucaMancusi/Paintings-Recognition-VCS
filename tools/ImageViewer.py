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
        try:
            if self.img_num <= self.cols:
                self.axarr[y].imshow(img)
                self.axarr[y].set_title(title)
            else:
                self.axarr[x, y].imshow(img)
                self.axarr[x, y].set_title(title)
        except IndexError:
            print('Failed to access the plot [{}, {}]'.format(x, y))

    
    def remove_axis_values(self):
        for plot in self.axarr.flatten():
            plot.set_yticklabels([])
            plot.set_xticklabels([])

    def __len__(self):
        return self.img_num
    
    def range(self):
        return range(len(self))

    def show(self):
        plt.show()

if __name__ == "__main__":
    from skimage import data
    im = data.coins()

    iv = ImageViewer(9, cols=3)
    iv.remove_axis_values()
    for i in iv.range():
        iv.add(im)
    iv.show() 