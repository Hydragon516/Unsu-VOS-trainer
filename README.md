# Unsu-VOS-trainer
unsupervised video object segmentation dataset &amp; train code

### Option
Edit options in config.py

### Dataset
We provide all dataset [here](https://drive.google.com/file/d/1WxIUs7yzgpJ7QOLXdU1xoYZZ2jysQ6bA/view?usp=sharing)

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
