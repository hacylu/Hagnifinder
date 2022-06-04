# Hagnifinder: Predicting Magnification of Digital Histological Images using Deep Learning in PyTorch

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

# Training 

To train the Hagnifinder, cd into this repo's `src` root folder and execute:

    $ python train.py


The script takes the following command line options:

- `root`: the root directory where tha dataset is stored, default to `'..\\dataset'`

- `epoch`: number of epochs to train for, default to `100`

- `batch_size`: The number of samples per training, default to `32`

- `num_workers`: default to `0`

- `N`: Threshold used to judge prediction results, default to `1`

- `data_arg`: data augmentation, set to True when training, set to False when testing. default to `True`

- `ASM`: Whether to use ASM  in the model, True represents yes, False represents no. default to `True`

- `ev`: expected variance in ASM. default to `2.2`

- `cs`: Calculate the scaling factor, set to True when using (ASM must also be set to True). default to `False`

- `sf`: Scaling factor, which takes effect when ASM=False. default to `0`

Learning rate, optimizer and loss function can all be set in __main__ of train.py.

### training strategy
1. Train with the parameter 'ASM'=True, the trained model will be saved in Hagnifinder/model_save/, and the model can be loaded through the load_model function in train.py.
2. Set the parameter 'cs'=True, , run 1epoch to calculate the scaling factor of the ASM in the current model (requires the parameter 'ASM'=True).
3. Set the parameter 'sf'= (Scaling factor calculated in step 2), continue to train the model (requires parameter 'ASM'=False, parameter 'cs'=False).

#### The flow chart of model training
![img.png](\img.png)

There is also a test function in train.py to test the model, change the code in the __main__ to test the model.




