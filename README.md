# Computer vision and cognitive approachesin the Galleria Estense Museum
### Authors: Gianluca Mancusi, Daniele Manicardi, Vittorio Pippi

The project aims to provide an application capable of processing paintings in images and videos, taken from the Galleria Estense in Modena.

### Installation

1.  (optional for the retrieval) Save a copy of the `paintings_db` in the following directory: `dataset\paintings_db`

2.  Download the weights of YOLO from here:
    https://pjreddie.com/media/files/yolov3.weights
    and save the file in the `yolo` directory.

3.  You have to install PyTorch and `torchvision`.

    A Windows only example of how to install PyTorch:
    ```sh
    pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

4.  Install the requirements.txt in your virtual environment
    ```sh
    pip install -r requirements.txt
    ```

5.  (optional) Download the weights of the U-Net and save them wherever you want. 
    You need to log in with the institutional account (UNIMORE).
    From this URL: https://drive.google.com/drive/u/2/folders/1J1imEqytdpz8P9lT2gBuB75a2rnP6HDo

### How to test the project
To start the project just run the python `gui.py` file, which will launch a web GUI from which you can test a pre-packaged image file or you can upload a new image or video file and test it.

Please note: the video output will be in `uploads\videos\outputs`
It is better not to compute too big (40MB) files or very long video.

##### U-Net test:
Run `predict.py` in the U-Net directory and give the weights you would like to test, the input and the output filename.
`predict.py --model MODEL.pth --input IMAGE_FILENAME`
