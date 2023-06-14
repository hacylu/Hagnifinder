# Hagnifinder-PyTorch (V1.1 was upgraded on 2023.06.14)

PyTorch implementation of ["Hagnifinder: Recovering Magnification Information of Digital Histological Images using Deep Learning"](https://www.sciencedirect.com/science/article/pii/S2153353923001165).
## Hagnifinder

we developed a regression model based on convolutional neural network (CNN) to accurately predict magnification of a given histology image, named Histology image magnification finder (Hagnifinder). The Hagnifinder proposed in this work could provide such a functionality: given a random snapshot with a random organ site, providing a predictive magnification level associate with the snapshot..

## Hagni40 Dataset

The dataset contains 94643 patches belonging to 40 different magnification levels and 13 different cancer types.

# Setup
### Requirements
Listed in `requirements.txt`. Install with `pip install -r requirements.txt` preferably in a virtualenv.

### Data
Place the dataset in the following structure：
```
Hagnifinder/
    Hagni40/
```

After this run `src/set_dataset_info.py` to load the dataset information.

# Test
#### To ensure prediction accuracy, make sure that the image being tested contains a large number of nuclei!!!

Predict histopathology images magnification using Hagnifinder:

    $ python test.py

The script takes the following command line options:

- `image_path`: The image path, default to '..\\img\\Laryngeal (1).png'

Citation:

1.  ZHANG H, LIU Z, SONG M, LU C. Hagnifinder: Recovering magnification information of digital histological images using deep learning[J/OL]. Journal of Pathology Informatics, 2023, 14: 100302. DOI:10.1016/j.jpi.2023.100302.

