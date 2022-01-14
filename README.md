# Assessment framework for Deepfake detection method

## Install & Requirements
The code has been tested on pytorch=1.9.1 and python 3.6, please refer to `requirements.txt` for more details.
### To install the python packages
`python -m pip install -r requirements.txt`

If you have problem installing dlib, please install it via `conda install -c conda-forge dlib`

### Videos download
If you would like to download the FaceForensics++, please go to this [page](https://github.com/ondyari/FaceForensics/tree/master/dataset) and fill the form the author provided. 

If you would like to access the Celeb-DF dataset, please go to this [page](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v1) and fill out the form the author provided.

### Frame extraction
For FF++, you can extract the frames of each videos using  `extract_frame_face_ff.py`

For Celeb-DF, you can extract the frames of each videos using  `extract_frame_face_celeb.py`

Example: `python extract_frame_face_ff.py --datapath <xxx> -d Deepfakes -c c23`

### Data set spilt 
For FF++, you can split the dataset through `split_ff.py` .  

For Celeb-DF you can split dataset through `split_celeb.py`. 
In these two files, you need to manually set paths and merge different txt files.

### Pretrained Model
We train our baseline models and improved models based on a pre-traind model.
- xception-b5690688.pth
### Perturbation
To add perturbation to FF++, you can use `random_distortion_ff.py` or use 4 sepearate python files: `Gamma_corr.py`, `Gau_blur.py`, `Gau_noise.py`, `jpeg_compression.py`

To add perturbation to Celeb-DF, you can use `random_distortion_celeb.py` or use 4 sepearate python files: `Gamma_corr_celeb.py`, `Gau_blur_celeb.py`, `Gau_noise_celeb.py`, `jpeg_compression_celeb.py`

Example: `python Gamma_corr.py --datapath <xxx> -d original -c raw -g 2.0`

### To train a model
`python train_CNN.py`
(Please set the arguments after read the code)

### To test the model on images
`python test_CNN.py`
(Please set the arguments after read the code)
