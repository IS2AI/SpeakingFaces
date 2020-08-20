# encoding: utf-8
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchaudio
import csv, pdb


class MyDataset(Dataset):
    def __init__(self, opt, data_type='train', noise=False):
        if data_type.lower() == 'train':
            self.data_path = opt.train_path
        elif data_type.lower() == 'valid':
            self.data_path = opt.valid_path
        elif data_type.lower() == 'test':
            self.data_path = opt.test_path

        self.opt = opt
        self.noise = noise
        if self.noise:
            if opt.mode in [1,4,5,7] and opt.add_rgb_noise:       #rgb images
                print("Adding {} noise to visual data (STD {})".format(opt.rgb_noise, opt.rnoise_value))
            if opt.mode in [2,4,6,7] and opt.add_thr_noise:       #rgb images
                print("Adding {} noise to thermal data (STD {})".format(opt.thr_noise, opt.tnoise_value))
            if opt.mode in [3,5,6,7] and opt.add_audio_noise:       #rgb images
                print("Adding {} noise to audio data (STD {})".format(opt.audio_noise, opt.anoise_value))

        self.sub_path       = opt.sub_path
        self.num_frames     = opt.num_frames
        self.mode           = opt.mode
        self.sr             = opt.sample_rate
        self.segment_len    = opt.segment_len
 
        #Read labels
        self.sub_label  = {}
        with open(self.sub_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if opt.predict.lower() == "age":
                    if int(row[2]) <= 24:
                        self.sub_label[row[0]] = 0
                    elif int(row[2]) <= 34:
                        self.sub_label[row[0]] = 1
                    else:
                        self.sub_label[row[0]] = 2
                elif opt.predict.lower() == "gender":
                    if row[3].lower() == 'female':
                        self.sub_label[row[0]] = float(0)
                    else:
                        self.sub_label[row[0]] = float(1)
                else:
                    print("Incorrect prediction value is given! TERMINATING...")
                    exit()

        subjects = os.listdir(self.data_path)
        subjects = list(filter(lambda sub: sub.find('sub_') != -1, subjects))
        subjects = sorted(subjects, key=lambda sub: int(sub.split('_')[1]))

        #Read input data features
        self.data = []
        for sub in subjects:
            label = self.sub_label[sub.split('_')[1]]

            trials = os.listdir(os.path.join(self.data_path,sub))
            trials = list(filter(lambda trial: trial.find('trial_') != -1, trials))
            trials = sorted(trials, key=lambda trial: int(trial.split('_')[1]))
            assert len(trials) == 2, "Incorrect number of trials: "+sub
            for trial in trials:
                #Use only trimmed utterance recordings
                records = os.listdir(os.path.join(self.data_path,sub,trial,"mic1_audio_cmd_trim"))
                records = list(filter(lambda record: record.find('.wav') != -1, records))
                records = sorted(records, key=lambda record: int(record.split('_')[4]))

                cmds = [record.split('_')[4] for record in records]

                if self.mode in [1,4,5,7]:  #rgb images
                    if not os.path.exists(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned")):
                        print(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned"))
                        continue
                    rgb_images = os.listdir(os.path.join(self.data_path,sub,trial,"rgb_image_cmd_aligned"))
                    rgb_images = list(filter(lambda image: image.find('_3.png') != -1, rgb_images))

                if self.mode in [2,4,6,7]:   #thermal images
                    if not os.path.exists(os.path.join(self.data_path,sub,trial,"thr_image_cmd")):
                        print(os.path.join(self.data_path,sub,trial,"thr_image_cmd"))
                        continue
                    thr_images = os.listdir(os.path.join(self.data_path,sub,trial,"thr_image_cmd"))
                    thr_images = list(filter(lambda image: image.find('_1.png') != -1, thr_images))

                for record in records:
                    cmd = record.split('_')[4]

                    if self.mode in [1,4,5,7]:       #rgb images
                        rgb_images_tmp = list(filter(lambda image: image.split('_')[4] == cmd, rgb_images))
                        rgb_images_tmp = sorted(rgb_images_tmp, key=lambda image: int(image.split('_')[5]))
                        if len(rgb_images_tmp) < self.num_frames:
                            print("This record has insufficient number of rgb frames: "+record)
                            continue
                        #downsample image stream
                        rgb_images_tmp = [rgb_images_tmp[i] for i in np.linspace(0,len(rgb_images_tmp)-1,
                                            endpoint=True,num=self.num_frames,dtype=int).tolist()]

                        #read images into an array list
                        rgb_array = [cv2.imread(os.path.join(self.data_path,sub,trial,
                                        "rgb_image_cmd_aligned",image)) for image in rgb_images_tmp]
                        rgb_array = list(filter(lambda image: not image is None, rgb_array))

                        #add noise (time consuming)
                        if self.noise and opt.add_rgb_noise:
                            rgb_array = [self.add_image_noise(image, opt.rgb_noise, opt.rnoise_value)
                                            for image in rgb_array]

                        #reduce image dimension
                        rgb_array = [cv2.resize(image, (116, 87), interpolation=cv2.INTER_LANCZOS4)
                                        for image in rgb_array]

                        #convert array list into numpy
                        rgb_array = np.stack(rgb_array, axis=0).astype(np.float32)

                    if self.mode in [2,4,6,7]:     #thermal images
                        thr_images_tmp = list(filter(lambda image: image.split('_')[4] == cmd, thr_images))
                        thr_images_tmp = sorted(thr_images_tmp, key=lambda image: int(image.split('_')[5]))
                        if len(thr_images_tmp) < self.num_frames:
                            print("This record has insufficient number of thermal frames: "+record)
                            continue
                        
                        #downsample image stream
                        thr_images_tmp = [thr_images_tmp[i] for i in np.linspace(0,len(thr_images_tmp)-1,
                                            endpoint=True,num=self.num_frames,dtype=int).tolist()]

                        #read images into an array list
                        thr_array = [cv2.imread(os.path.join(self.data_path,sub,trial,
                                        "thr_image_cmd",image)) for image in thr_images_tmp]
                        thr_array = list(filter(lambda image: not image is None, thr_array))

                        #add noise (time consuming)
                        if self.noise and opt.add_thr_noise:
                            thr_array = [self.add_image_noise(image, opt.thr_noise, opt.tnoise_value)
                                            for image in thr_array]

                        #reduce image dimension
                        thr_array = [cv2.resize(image, (116, 87), interpolation=cv2.INTER_LANCZOS4)
                                        for image in thr_array]

                        #convert array list into numpy
                        thr_array = np.stack(thr_array, axis=0).astype(np.float32)

                    if self.mode in [3,5,6,7]:
                        audio, samplerate = torchaudio.load(os.path.join(self.data_path, sub,
                                                        trial, 'mic1_audio_cmd_trim', record))
                        audio = audio.squeeze()
                        
                        if self.noise and opt.add_audio_noise:
                            #add additive white Gaussian noise (AWGN)
                            audio    = self.add_audio_noise(audio, opt.audio_noise, opt.anoise_value)

                        if len(audio) < self.segment_len*self.sr:
                            print("This record has insufficient number of audio features: "+record)
                            continue
                        
                        #extract audio snippet from the middle of record
                        audio = audio[len(audio)//2-int(self.segment_len*self.sr)//2:
                                      len(audio)//2+int(self.segment_len*self.sr)//2]

                        #Compute  mel spectogram features
                        spec = torchaudio.transforms.MelSpectrogram()(audio)
                        spec = spec.transpose(1,0) # (Feature, Time) -> (Time, Feature)
                        spec = spec.unsqueeze(2)
                        spec = spec.unsqueeze(3)

                    if self.mode == 1:      #rgb images
                        self.data.append([rgb_array, label, sub])
                    elif self.mode == 2:    #thermal images
                        self.data.append([thr_array, label, sub])
                    elif self.mode == 3:    #audio
                        self.data.append([spec, label, sub])
                    elif self.mode == 4:    #rgb and thermal
                        self.data.append([rgb_array, thr_array, label, sub])
                    elif self.mode == 5:    #rgb and thermal
                        self.data.append([rgb_array, spec, label, sub])
                    elif self.mode == 6:    #thermal and audio
                        self.data.append([thr_array, spec, label, sub])
                    elif self.mode == 7:    #rgb, thermal and audio
                        self.data.append([rgb_array, thr_array, spec, label, sub])

    def add_image_noise(self, image, noise_type='gauss', noise_value=10):
        if noise_type.lower() == "gauss":
            row,col,ch = image.shape
            mean = 0
            sigma = noise_value
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy_image = image + gauss
        elif noise_type.lower() == "blur":
            kernel_size = (noise_value, noise_value)
            noisy_image = cv2.blur(image, kernel_size)
        else:
            print("Incorrect image noise type is given! Terminating ...")
            exit()

        return noisy_image.astype("uint8")

    def add_audio_noise(self, audio, noise_type='gauss', noise_value=10):
        if noise_type.lower() == "gauss":
            mean = 0
            sigma = noise_value
            gauss = torch.normal(mean, sigma, size=(len(audio),))
            noisy_audio = audio + gauss
        else:
            print("Incorrect audio noise type is given! Terminating ...")
            exit()

        return noisy_audio 

    def __getitem__(self, idx):
        if self.mode == 1:       #rgb images
            (rgb_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'label': label,
                    'sub': sub}
        elif self.mode == 2:     #thermal images
            (thr_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'label': label,
                    'sub': sub}
        elif self.mode == 3:     #audio
            (audio, label, sub) = self.data[idx]
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
        elif self.mode == 4:       #rgb images
            (rgb_images, thr_images, label, sub) = self.data[idx]
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'label': label,
                    'sub': sub}
        elif self.mode == 5:
            (rgb_images, audio, label, sub) = self.data[idx]
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
        elif self.mode == 6:
            (thr_images, audio, label, sub) = self.data[idx]
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}
        elif self.mode == 7:
            (rgb_images, thr_images, audio, label, sub) = self.data[idx]
            audio = F.pad(audio,(1,1),"constant", 0)
            # (T, H, W, C)->(C, T, H, W)
            return {'rgb_images': torch.FloatTensor(rgb_images.transpose(3, 0, 1, 2)), 
                    'thr_images': torch.FloatTensor(thr_images.transpose(3, 0, 1, 2)),
                    'audio': audio.permute(3, 0, 1, 2),
                    'label': label,
                    'sub': sub}

    def __len__(self):
        return len(self.data)
