## Thermal-Visible Image Translation
### Dataset
Download [thermal2visible_speakingfaces](https://drive.google.com/drive/folders/1tMFXKaoy1EkJdafi3VpKZa6_ovUS4OJM?usp=sharing) dataset. 

### CycleGAN Model
1. Clone the original repo https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and install dependencies. 
2. Train the model on thermal2visible_speakingfaces dataset: 
```
python train.py  --dataroot ./thermal2visible_speakingfaces/ --model cycle_gan --gpu_ids 0 --batch_size 1 --display_id 0 --n_epochs 50 --n_epochs_decay 50 --lr 0.0001 --beta1 0.1 --netG resnet_9blocks --load_size 130 --crop_size 128 --name gan_cycle 
```
3. Test the model on thermal2visible_speakingfaces dataset:
```
python test.py  --dataroot ./thermal2visible_speakingfaces/  --model cycle_gan --gpu_ids 0 --batch_size 1 --load_size 130 --crop_size 128 --name gan_cycle --num_test 2268
```

### CUT Model
1. Clone the original repo https://github.com/taesungp/contrastive-unpaired-translation and install dependencies. 
2. Train the model on thermal2visible_speakingfaces dataset: 
```
python train.py  --dataroot ./thermal2visible_speakingfaces/  --name gan_cut --CUT_mode CUT --gpu_ids 0 --batch_size 2 --display_id 0 --n_epochs 200 --n_epochs_decay 200 --lr 0.0002 --netG resnet_9blocks --load_size 130 --crop_size 128 
```
3. Test the model on thermal2visible_speakingfaces dataset:
```
python test.py  --dataroot ./thermal2visible_speakingfaces/  --CUT_mode CUT --gpu_ids 0 --batch_size 1  --netG resnet_9blocks --load_size 258 --crop_size 256 --name gan_cut --num_test 2268
```
## Face Recognition 
1. Download our results on the test set: [GANs_results.zip](https://drive.google.com/drive/folders/1tMFXKaoy1EkJdafi3VpKZa6_ovUS4OJM?usp=sharing).  
2. Extract and save the facial embeddings from the real visible images (test set, only first trial):
```
python embeddings.py --images ./GANs_results/real_B
```
3. Test the face recognition on the real visible images (test set, second trial):
```
python face_recognition_test.py --images ./GANs_results/real_B/ --thr 0.45 
```
4. Test the face recognition on the real thermal images (test set, second trial):
```
python face_recognition_test.py --images ./GANs_results/real_A/ --thr 0.45 
```
5. Test the face recognition on the fake visible images (test set, second trial, CycleGAN):
```
python face_recognition_test.py --images ./GANs_results/fake_B_cycle_gan/ --thr 0.45 --size 112
```
6. Test the face recognition on the fake visible images (test set, second trial, CUT):
```
python face_recognition_test.py --images ./GANs_results/fake_B_cut/ --thr 0.45 --size 112
```

## If you use visible2thermal_speakingfaces dataset and/or the provided code in your research then please cite our paper:
```
@misc{abdrakhmanova2020speakingfaces,
      title={SpeakingFaces: A Large-Scale Multimodal Dataset of Voice Commands with Visual and Thermal Video Streams}, 
      author={Madina Abdrakhmanova and Askat Kuzdeuov and Sheikh Jarju and Yerbolat Khassanov and Michael Lewis and Huseyin Atakan Varol},
      year={2020},
      eprint={2012.02961},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
