# Dependencies
* Theano (sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git@5159a6b3cc2875f90e86449e5b3914f7bcb76452)
* Lasagne
* SciPy
* Numpy
* bcolz

# Setup paths
Set your custom paths in ```paths.py```.

# Preprocessing the data
This takes more or less 150GB.

```python scripts/videos_prep.py```

# Training the model
This is not required as the pretrained models are included.
 
```python train.py --jobs=4 --config=models/3drnn_sep2d_elu_nod_mlr.py```

# Generating predictions + submission

```python predict.py --set=test --meta=metadata/3drnn_sep2d_elu_nod_mlr-hond-20170620-115014.pkl```

```python submission.py --set=test --meta=metadata/3drnn_sep2d_elu_nod_mlr-hond-20170620-115014.pkl```

Replace "test" to "valid" to make a submission file for the validation set.
If the models are retrained, replace the metadata files to the new ones.

