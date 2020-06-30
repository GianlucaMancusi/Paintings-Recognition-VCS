# Computer vision and cognitive approachesin the Galleria Estense Museum
### Authors: Gianluca Mancusi, Daniele Manicardi, Vittorio Pippi

The project aims to provide an application capable of processing paintings in images and videos, taken from the Galleria Estense in Modena.

### Installation

Install the requirements.txt in your virtual environment
```sh
pip install -r requirements.txt
```

Save a copy of the `paintings_db` in the following directory: `dataset\paintings_db`

Download the weights of YOLO and save the file in the `yolo` directory.
```sh
cd yolo
wget https://pjreddie.com/media/files/yolov3.weights
```

Download the weights of the U-Net. You need to log in with the institutional account (UNIMORE).
From this URL: https://drive.google.com/drive/u/2/folders/1J1imEqytdpz8P9lT2gBuB75a2rnP6HDo

### How to test the project
To start the project just run the python `gui.py` file, which will launch a web GUI from which you can test a pre-packaged image file or you can upload a new image or video file and test it.

Please note: for videos it is better not to upload too big (>20MB) or very long files.

##### U-Net test:
Run `predict.py` in the U-Net directory and give the weights you would like to test, the input and the output filename.
`predict.py --model MODEL.pth --input `
