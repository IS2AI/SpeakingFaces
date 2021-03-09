## Thermal-Visible Image Translation
### [Dataset](https://drive.google.com/drive/folders/1tMFXKaoy1EkJdafi3VpKZa6_ovUS4OJM?usp=sharing)
Download thermal2visible_speakingfaces dataset. 

### CycleGAN
1. Clone the original repo https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and install dependencies. 
2. Train the model on thermal2visible_speakingfaces dataset: 
```
python train.py  --dataroot ./thermal2visible_speakingfaces/ --model cycle_gan --gpu_ids 0 --batch_size 1 --display_id 0 --n_epochs 50 --n_epochs_decay 50 --lr 0.0001 --beta1 0.1 --netG resnet_9blocks --load_size 130 --crop_size 128 --name gan_cycle 
```
3. Test the model on thermal2visible_speakingfaces dataset:
```
python test.py  --dataroot ./thermal2visible_speakingfaces/  --model cycle_gan --gpu_ids 0 --batch_size 1 --load_size 130 --crop_size 128 --name gan_cycle --num_test 1134
```

### CUT
1. Clone the original repo https://github.com/taesungp/contrastive-unpaired-translation and install dependencies. 
2. Train the model on thermal2visible_speakingfaces dataset: 
```
python train.py  --dataroot ./thermal2visible_speakingfaces/  --name gan_cut --CUT_mode CUT --gpu_ids 0 --batch_size 2 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --lr 0.0002 --netG resnet_9blocks --load_size 130 --crop_size 128 
```
3. Test the model on thermal2visible_speakingfaces dataset:
```
python test.py  --dataroot ./thermal2visible_speakingfaces/  --CUT_mode CUT --gpu_ids 0 --batch_size 1  --netG resnet_9blocks --load_size 258 --crop_size 256 --name gan_cut --num_test 1134
```
