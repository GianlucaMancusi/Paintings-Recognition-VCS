# Computer vision and cognitive approachesin the Galleria Estense Museum
### Authors: Gianluca Mancusi, Daniele Manicardi, Vittorio Pippi

The  project  aims  to  provide  an  application  capable  of processing artistic images and videos, taken from the Galleria Estense in Modena.

### Installation
Download the weights of YOLO.
```sh
cd yolo
wget https://pjreddie.com/media/files/yolov3.weights
```

Download the weights of U-Net.
```sh
cd Pytorch-UNet-master
wget wget "https://drive.google.com/uc?export=download&id=ID_FILE_UNET"
```


### How to test the project
To start the project just start the python `gui.py` file, which will launch a web GUI from which you can test a sample file and you can upload a new image or video file and test it.

Please note: for videos it is better not to upload too big or very long files.