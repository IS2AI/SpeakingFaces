import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import SFNet
import numpy as np
import os, time, pdb


if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=opt.data_shuffle):
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

def test(model, loader, dataset, test=False):
    with torch.no_grad():
        tic = time.time()
        model.eval()

        if opt.predict.lower() == "gender":
            loss_fn = nn.BCEWithLogitsLoss()
        elif opt.predict.lower() == "age":
            loss_fn = nn.CrossEntropyLoss()
        else:
            print("Incorrect prediction task is given! TERMINATING...")
            exit()

        correct = 0
        counter = 0
        total_loss = 0
        for (i_iter, input) in enumerate(loader):
            label = input.get('label').cuda()
            if opt.mode == 1:   #rgb
                rgb_images  = input.get('rgb_images').cuda()
                sub         = input.get('sub')
                output      = model(x1=rgb_images)
                counter     += len(rgb_images)
            elif opt.mode == 2: #thermal
                thr_images  = input.get('thr_images').cuda()
                sub         = input.get('sub')
                output      = model(x2=thr_images)
                counter     += len(thr_images)
            elif opt.mode == 3: #audio
                audio       = input.get('audio').cuda()
                sub         = input.get('sub')
                output      = model(x3=audio)
                counter     += len(audio)
            elif opt.mode == 4: #rgb+thermal
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                sub         = input.get('sub')
                output      = model(x1=rgb_images, x2=thr_images)
                counter     += len(rgb_images)
            elif opt.mode == 5: #rgb+audio
                rgb_images  = input.get('rgb_images').cuda()
                audio       = input.get('audio').cuda()
                sub         = input.get('sub')
                output      = model(x1=rgb_images,x3=audio)
                counter     += len(rgb_images)
            elif opt.mode == 6: #thermal+audio
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                sub         = input.get('sub')
                output      = model(x2=thr_images,x3=audio)
                counter     += len(thr_images)
            elif opt.mode == 7: #rgb+thermal+audio
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                sub         = input.get('sub')
                output      = model(x1=rgb_images, x2=thr_images, x3=audio)
                counter     += len(rgb_images)
 
            if opt.predict.lower() == "gender":
                loss = loss_fn(output.view(-1), label)
            elif opt.predict.lower() == "age":
                loss = loss_fn(output, label)
            total_loss += loss.item()

            if opt.predict.lower() == "gender":
                output = torch.sigmoid(output)
                output = (output>0.5).float()
                correct += (output.view(-1) == label).float().sum()
            elif opt.predict.lower() == "age":
                max_index = output.max(dim = 1)[1]
                correct += (max_index == label).float().sum()

            if opt.print_errors and test:
                for i, s in enumerate(zip(output, label)):
                    if s[0] != s[1]:
                        print("Incorrect output: {}".format(sub[i]))

        accuracy = 100*correct/len(dataset)

        return (total_loss, accuracy, (time.time()-tic)/60)

def train(model):
    #Construct model savename
    savename = ('{0:}_mode{1:}_lr{2:}_wd{3:}_patience{4:}_drop{5:}_epoch{6:}_bs{7:}_seed{8:}_clip{9:}').format(
                    opt.save_prefix, opt.mode, opt.base_lr, opt.weight_decay, opt.patience, opt.drop,
                    opt.max_epoch, opt.batch_size, opt.random_seed, opt.clip)

    if opt.mode in [1,4,5,7]:
        savename += "_rfr"+str(opt.num_frames)
        if opt.add_rgb_noise:
            savename += "_rnoise"+opt.rgb_noise+"_rnvalue"+str(opt.rnoise_value)

    if opt.mode in [2,4,6,7]:
        savename += "_tfr"+str(opt.num_frames)
        if opt.add_thr_noise:
            savename += "_tnoise"+opt.thr_noise+"_tnvalue"+str(opt.tnoise_value)

    if opt.mode in [3,5,6,7]:
        savename += "_len"+str(opt.segment_len)
        if opt.add_audio_noise:
            savename += "_anoise"+opt.audio_noise+"_anvalue"+str(opt.anoise_value)

    #Create a folder where models will be saved
    (path, name) = os.path.split(savename)
    if(not os.path.exists(path)):
        os.makedirs(path)

    #Load data
    print("Loading data...")
    dataset = MyDataset(opt,'train',noise=opt.train_noise)
    loader  = dataset2dataloader(dataset, shuffle=False)
    print('Training data size: {}'.format(len(dataset)))

    valid_dataset = MyDataset(opt,'valid',noise=opt.valid_noise)
    valid_loader = dataset2dataloader(valid_dataset, shuffle=False)
    print('Validation data size: {}'.format(len(valid_dataset)))

    test_dataset = MyDataset(opt,'test',noise=opt.test_noise)
    test_loader = dataset2dataloader(test_dataset, shuffle=False)
    print('Test data size: {}'.format(len(test_dataset)))

    #Setup optimizer and scheduler
    if opt.warmup_epochs > 0:
        warmup = True
        savename += "_warmuplr"+str(opt.warmup_lr)+"_warmupepochs"+str(opt.warmup_epochs)
        optimizer = optim.Adadelta(model.parameters(), lr=opt.warmup_lr)
    else:
        warmup = False
        #optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay, amsgrad=True)
        optimizer = optim.Adadelta(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay)
        #optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay,
        #                      momentum=0.9,nesterov=True)

        #scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                        patience=opt.patience, verbose=True, threshold=1e-4)

    #Setup loss function
    if opt.predict.lower() == "gender":
        loss_fn = nn.BCEWithLogitsLoss()
    elif opt.predict.lower() == "age":
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("Incorrect prediction value '{}' is given! TERMINATING...")
        exit()

    #Start training process
    tic = time.time()
    best_acc = 0
    best_epoch = 0
    for epoch in range(opt.max_epoch+opt.warmup_epochs):
        tic_epoch = time.time()
        model.train()
        if warmup and epoch >= opt.warmup_epochs:
            print("\nWARMUP IS COMPLETE!")
            warmup = False
            optimizer = optim.Adadelta(model.parameters(), lr=opt.base_lr, weight_decay=opt.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                            patience=opt.patience, verbose=True, threshold=1e-4)

        correct = 0
        counter = 0
        total_loss = 0
        for (i_iter, input) in enumerate(loader):
            label = input.get('label').cuda()
            optimizer.zero_grad()
            if opt.mode == 1:   #rgb
                rgb_images  = input.get('rgb_images').cuda()
                output      = model(x1=rgb_images)
                counter     += len(rgb_images)
            elif opt.mode == 2: #thermal
                thr_images  = input.get('thr_images').cuda()
                output      = model(x2=thr_images)
                counter     += len(thr_images)
            elif opt.mode == 3: #audio
                audio       = input.get('audio').cuda()
                output      = model(x3=audio)
                counter     += len(audio)
            elif opt.mode == 4: #rgb+thermal
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                output      = model(x1=rgb_images,x2=thr_images)
                counter     += len(rgb_images)
            elif opt.mode == 5: #rgb+audio
                rgb_images  = input.get('rgb_images').cuda()
                audio       = input.get('audio').cuda()
                output      = model(x1=rgb_images, x3=audio)
                counter     += len(rgb_images)
            elif opt.mode == 6: #thermal+audio
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                output      = model(x2=thr_images, x3=audio)
                counter     += len(thr_images)
            elif opt.mode == 7: #rgb+thermal+audio
                rgb_images  = input.get('rgb_images').cuda()
                thr_images  = input.get('thr_images').cuda()
                audio       = input.get('audio').cuda()
                output      = model(x1=rgb_images, x2=thr_images, x3=audio)
                counter     += len(rgb_images)

            if opt.predict.lower() == "gender":
                loss = loss_fn(output.view(-1), label)
            elif opt.predict.lower() == "age":
                loss = loss_fn(output, label)

            total_loss += loss.item()
            loss.backward()

            if opt.is_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

            optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            if opt.predict.lower() == "gender":
                output = torch.sigmoid(output)
                output = (output>0.5).float()
                correct += (output.view(-1) == label).float().sum()
            elif opt.predict.lower() == "age":
                max_index = output.max(dim = 1)[1]
                correct += (max_index == label).float().sum()

        #Compute train set accuracy
        train_acc = 100*correct/len(dataset)
        print('\n' + ''.join(81*'*'))
        print('EPOCH={}, lr={}'.format(epoch, show_lr(optimizer)))
        print('TRAIN SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                    total_loss/len(dataset),(time.time()-tic_epoch)/60, train_acc))

        #Evaluate model on the validation set
        (valid_loss, valid_acc, valid_time) = test(model, valid_loader, valid_dataset)
        print('VALID SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                    valid_loss/len(valid_dataset), valid_time, valid_acc))
        print('Best valid acc={:.3f}, best epoch={}'.format(best_acc, best_epoch))
        tmp_savename = savename + "_bestEpoch"+str(best_epoch)+".torch"
        print("MODEL: {}".format(os.path.split(tmp_savename)[1]))
        print(''.join(81*'*'))
        if not warmup:
            scheduler.step(valid_loss)
 
        #Save the best model
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
            tmp_savename = savename + "_bestEpoch"+str(best_epoch)+".torch"
            print("Saving the best model (best acc={:.3f})".format(best_acc))
            torch.save(model.state_dict(), tmp_savename)

    print('\n' + ''.join(81*'*'))
    print("Total trianing time = {:.2f}m".format((time.time()-tic)/60))
    print("Best valid acc = {:.3f}".format(best_acc))
    print("MODEL: {}".format(os.path.split(tmp_savename)[1]))
    print(''.join(81*'*'))

    #Evaluate model on the test set
    print('\n' + ''.join(81*'*'))
    load_model(model, tmp_savename)
    (test_loss, test_acc, test_time) = test(model, test_loader, test_dataset, test=True)
    print('TEST SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                    test_loss/len(test_dataset), test_time, test_acc))
    print(''.join(81*'*'))

def load_model(model, path):
    print("\n"+"Loading model...")
    print("Model name: {}".format(path))
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('missmatched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if(__name__ == '__main__'):
    #Setup seed values and other options for experiment repeatability
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opt.random_seed)

    print("Building model...")
    model = SFNet(opt).cuda()

    if(hasattr(opt, 'weights')):
        #Load data
        print("\n"+"Loading valid data...")
        valid_dataset = MyDataset(opt,'valid',noise=opt.valid_noise)
        valid_loader = dataset2dataloader(valid_dataset, shuffle=False)
        print('Validation data size: {}'.format(len(valid_dataset)))

        print("\n"+"Loading test data...")
        test_dataset = MyDataset(opt,'test',noise=opt.test_noise)
        test_loader = dataset2dataloader(test_dataset, shuffle=False)
        print('Test data size: {}'.format(len(test_dataset)))

        if True:
            #Evaluate a single model
            #Load saved model
            load_model(model, opt.weights)

            #Evaluate model on the validatin and test sets
            print("\n"+"Evaluating...")
            print(''.join(81*'*'))
            (valid_loss, valid_acc, valid_time) = test(model, valid_loader, valid_dataset, test=True)
            print('VALID SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                        valid_loss/len(valid_dataset), valid_time, valid_acc))
            print(''.join(81*'*'))
            (test_loss, test_acc, test_time) = test(model, test_loader, test_dataset, test=True)
            print('TEST SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                        test_loss/len(test_dataset), test_time, test_acc))
            print(''.join(81*'*'))
        else:
            #Evaluate multiple models
            #mode1
            #gender
            #model_names=['models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.2_epoch200_mode1_seed0_clip10_rfr3_bestEpoch100.torch','models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.2_epoch200_mode1_seed1_clip10_rfr3_bestEpoch66.torch','models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.2_epoch200_mode1_seed2_clip10_rfr3_bestEpoch102.torch','models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.2_epoch200_mode1_seed3_clip10_rfr3_bestEpoch64.torch','models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.2_epoch200_mode1_seed4_clip10_rfr3_bestEpoch96.torch']
            #age
            #model_names=['models/mmnet_mode1_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed0_clip10_rfr3_bestEpoch136.torch','models/mmnet_mode1_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed1_clip10_rfr3_bestEpoch124.torch','models/mmnet_mode1_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed2_clip10_rfr3_bestEpoch179.torch','models/mmnet_mode1_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed3_clip10_rfr3_bestEpoch199.torch','models/mmnet_mode1_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed4_clip10_rfr3_bestEpoch140.torch']

            #mode2
            #gender
            #model_names=['models/mmnet_bs256_lr0.1_wd0.0_patience20_drop0.4_epoch200_mode2_seed0_clip10_tfr3_bestEpoch192.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed2_clip10_tfr3_bestEpoch177.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed3_clip10_tfr3_bestEpoch185.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed5_clip10_tfr3_bestEpoch198.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed7_clip10_tfr3_bestEpoch185.torch']
            #age
            #model_names=['models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed0_clip10_tfr3_bestEpoch30.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed1_clip10_tfr3_bestEpoch77.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed2_clip10_tfr3_bestEpoch28.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed3_clip10_tfr3_bestEpoch25.torch','models/mmnet_mode2_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed4_clip10_tfr3_bestEpoch60.torch']

            #mode3
            #gender
            #model_names=['models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed0_clip10_len0.4_bestEpoch134.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed1_clip10_len0.4_bestEpoch93.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed2_clip10_len0.4_bestEpoch119.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed3_clip10_len0.4_bestEpoch135.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed4_clip10_len0.4_bestEpoch107.torch']
            #age
            #model_names=['models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.0_epoch200_bs256_seed0_clip10_len0.4_bestEpoch24.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.0_epoch200_bs256_seed1_clip10_len0.4_bestEpoch17.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.0_epoch200_bs256_seed2_clip10_len0.4_bestEpoch29.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.0_epoch200_bs256_seed3_clip10_len0.4_bestEpoch27.torch','models/mmnet_mode3_lr0.1_wd0.0_patience20_drop0.0_epoch200_bs256_seed4_clip10_len0.4_bestEpoch21.torch']

            #mode4
            #gender
            #model_names=['models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed0_clip10_rfr3_tfr3_bestEpoch121.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed1_clip10_rfr3_tfr3_bestEpoch52.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed2_clip10_rfr3_tfr3_bestEpoch74.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed3_clip10_rfr3_tfr3_bestEpoch89.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed4_clip10_rfr3_tfr3_bestEpoch73.torch']
            #age
            #model_names=['models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed0_clip10_rfr3_tfr3_bestEpoch111.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed1_clip10_rfr3_tfr3_bestEpoch23.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed2_clip10_rfr3_tfr3_bestEpoch35.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed3_clip10_rfr3_tfr3_bestEpoch32.torch','models/mmnet_mode4_lr0.1_wd0.0_patience20_drop0.3_epoch200_bs256_seed4_clip10_rfr3_tfr3_bestEpoch36.torch']

            #mode5
            #gender
            #model_names=['models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed0_clip10_rfr3_len0.4_bestEpoch70.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed1_clip10_rfr3_len0.4_bestEpoch65.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed2_clip10_rfr3_len0.4_bestEpoch60.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed3_clip10_rfr3_len0.4_bestEpoch47.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.4_epoch200_bs256_seed4_clip10_rfr3_len0.4_bestEpoch71.torch']
            #age
            #model_names=['models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed0_clip10_rfr3_len0.4_bestEpoch104.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed1_clip10_rfr3_len0.4_bestEpoch179.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed2_clip10_rfr3_len0.4_bestEpoch123.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed3_clip10_rfr3_len0.4_bestEpoch117.torch','models/mmnet_mode5_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed4_clip10_rfr3_len0.4_bestEpoch156.torch']

            #mode6
            #gender
            #model_names=['models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed0_clip10_tfr3_len0.4_bestEpoch75.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed1_clip10_tfr3_len0.4_bestEpoch55.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed2_clip10_tfr3_len0.4_bestEpoch83.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed3_clip10_tfr3_len0.4_bestEpoch54.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed4_clip10_tfr3_len0.4_bestEpoch48.torch']
            #age
            #model_names=['models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed0_clip10_tfr3_len0.4_bestEpoch16.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed1_clip10_tfr3_len0.4_bestEpoch23.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed2_clip10_tfr3_len0.4_bestEpoch20.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed3_clip10_tfr3_len0.4_bestEpoch23.torch','models/mmnet_mode6_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed4_clip10_tfr3_len0.4_bestEpoch26.torch']

            #mode7
            #gender
            #model_names=['models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed0_clip10_rfr3_tfr3_len0.4_bestEpoch61.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed1_clip10_rfr3_tfr3_len0.4_bestEpoch63.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed2_clip10_rfr3_tfr3_len0.4_bestEpoch61.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed3_clip10_rfr3_tfr3_len0.4_bestEpoch63.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.1_epoch200_bs256_seed4_clip10_rfr3_tfr3_len0.4_bestEpoch65.torch']
            #age
            #model_names=['models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed0_clip10_rfr3_tfr3_len0.4_bestEpoch27.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed1_clip10_rfr3_tfr3_len0.4_bestEpoch25.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed2_clip10_rfr3_tfr3_len0.4_bestEpoch28.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed3_clip10_rfr3_tfr3_len0.4_bestEpoch23.torch','models/mmnet_mode7_lr0.1_wd0.0_patience20_drop0.2_epoch200_bs256_seed4_clip10_rfr3_tfr3_len0.4_bestEpoch23.torch']

            for m in model_names:
                load_model(model, m)
                print("\n"+"Evaluating...")
                #Evaluate model on the test set
                print(''.join(81*'*'))
                (valid_loss, valid_acc, valid_time) = test(model, valid_loader, valid_dataset, test=True)
                print('VALID SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                        valid_loss/len(valid_dataset), valid_time, valid_acc))
                print(''.join(81*'*'))
                (test_loss, test_acc, test_time) = test(model, test_loader, test_dataset, test=True)
                print('TEST SET: total loss={:.8f}, time={:.2f}m, acc={:.3f}'.format(
                        test_loss/len(test_dataset), test_time, test_acc))
                print(''.join(81*'*'))
            exit()

    else:
        train(model)

