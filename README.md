# Differential Diagnosis on Diffuse Interstitial Lung Disease by Quantifying Imaging Patterns with Multi-Tasks Deep Learning in High-Resolution CT of the Lungs

---

![Figure_main_fig](https://user-images.githubusercontent.com/46750574/189354334-7e4b8c10-71e7-48e2-97de-0548be2a68be.png)


## Directory Architecture

Root

|---------- README.md

|---------- train.json (You have to create yourself.)

|---------- valid.json (You have to create yourself.)

|---------- test.json (You have to create yourself.)

|---------- config.py

|---------- datasets.py

|---------- datasets_3D.py

|---------- train_segmentation.py # for slice-level task

|---------- train_classification_3D.py # for patient-level task

|---------- train_segmentation_3D.py # for patient-level task

|---------- test_majority_voting.py # for slice-level voting system

|---------- utils_folder.py

|---------- runs (if you run the train code, it will be made automatically)

|---------- checkpoints (if you run the train code, it will be made automatically)

## Train (Slice-level task)

```
CUDA_VISIBLE_DEVICES=0 python train_segmentation.py \
--msg=3_class_aux_dice_COP_oversampling \
--print_freq=10 --w 6 --batch_size 10 --img_size 512 --num_class 3 \
--backbone unet --seg_loss dice --aug True --naive_lung_mul True \
--lambda_seg 1 --lambda_cls 1 \
--train_path ./json/train.json \
--test_path ./json/test.json \
--val_path ./json/valid.json
```

## Train (3D classification model)

```
CUDA_VISIBLE_DEVICES=1 python train_classification_3D.py \ 
--msg=3_class_3D_resnet34 \
--print_freq=10 --w 6 --batch_size 1 --img_size 384 \
--num_class 3 --backbone resnet --aug True --lung_mul True \
--train_path ./json/train.json \
--test_path ./json/test.json \
--val_path ./json/val.json
```

## Train (3D segmentation multi-task learning model)

```
CUDA_VISIBLE_DEVICES=0 python train_segmentation_3D.py \
--msg=3_class_BCE_aug_3D_shallow_MTL \
--print_freq=10 --w 6 --batch_size 1 --img_size 384 --out_channels 6 \
--num_class 3 --backbone unet --seg_loss BCE --aug True \
--train_path ./json/train.json \
--test_path ./json/test.json \
--val_path ./json/val.json
```
## Train (2D Bi-LSTM model)

```
CUDA_VISIBLE_DEVICES=0 python train_bi_LSTM.py \
--root_dir ../pickle/ --num_workers 16 --additional_domain train 
--tag LSTM_Lung
```

## Test (Majority voting)

```
CUDA_VISIBLE_DEVICES=0 python test_majority_voting.py \
--msg=3_class_aux_unet_test \
--w 6 --img_size 512 --num_class 3 \
--backbone unet --naive_lung_mul True \
--test_path ./json/test.json \
--resume "your model checkpoint" \
```

## Test (2D Bi-LSTM model)

```
CUDA_VISIBLE_DEVICES=0 python test_bi_LSTM.py
```