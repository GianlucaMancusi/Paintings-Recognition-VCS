import tkinter as tk
from PIL import ImageTk,Image 

window = tk.Tk()
window.geometry('1580x720')
window.title('Project')
window.grid_columnconfigure(0, weight=3)
window.grid_columnconfigure(1, weight=1)

canvas = tk.Canvas(window)
canvas.pack()
canvas.grid(row=0, column=0)
img = ImageTk.PhotoImage(Image.open('data_test/persone.jpg'))
canvas.create_image(0,0, image=img)

if __name__ == "__main__":
    window.mainloop()