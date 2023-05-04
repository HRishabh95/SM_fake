# Social Media Fake News Detection

BERT Model training for Twitter Fake News Detection on CMU Misinfo, Ginger Cure Covid and FakeHealth Dataset.


## Read Dataset:

```
read_CMU.py -  Reading CMU Misinfo Dataset
read_ginger.py - Reading Ginger Cure Covid Dataset
read SM_1.py - Reading FakeHealth Dataset
```

## Spliting Dataset:

```
split_data.py - Spliting Dataset in 5 folds. 

Change the filepath for specific datasets.
```

## Training CE:

```
train_ce.py - Training Cross Encoder with folds files.
```