# Unsu-VOS-trainer
unsupervised video object segmentation dataset &amp; train code

### Option
Edit options in config.py

### Dataset


### STEP 1
Pretrain with DUTS dataset
```
python pretrain.py
```

### STEP 2
Fine-tuning with DAVIS dataset
```
python train_for_DAVIS.py
```

### Evaluation
```
python test_for_FBMS.py
```
or
```
python test_for_YTobj.py
```
