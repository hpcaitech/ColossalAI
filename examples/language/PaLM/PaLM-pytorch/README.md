# PaLM

## Download data
The enwik8 data was downloaded from the Hutter prize page: http://prize.hutter1.net/.
Then you need to use the `PATH` you choose to get the data in `train.py`:
```python
with gzip.open("PATH/enwik8.gz") as file:
```

## Run
You can use the following command to start training:
```python
python train.py
```
