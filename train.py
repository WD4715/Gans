import torch
from Network import SwinUnet
from dataloader import dataloader
import numpy as np
from utils import build_optimizer, build_scheduler, AttrDict, mp_batch
from torchvision.models import resnet50
import torch.nn as nn
from tqdm import tqdm
import wandb
import argparse

########################################################################################################################
########################################################################################################################
########################################################################################################################
############################################  TRAIN ####################################################################

config = AttrDict()
config.TRAIN = AttrDict()
config.TRAIN.USE_CHECKPOINT = False
config.TRAIN.WEIGHT_DECAY = 0.05
config.TRAIN.BASE_LR = 5e-4
config.TRAIN.MIN_LR = 5e-6
config.TRAIN.WARMUP_LR = 5e-7
config.TRAIN.CLIP_GRAD = 5.0
config.TRAIN.AUTO_RESUME = True
config.TRAIN.ACCUMULATION_STEPS = 5
config.TRAIN.EPOCHS = 100
config.TRAIN.WARMUP_EPOCHS = 20
config.TRAIN.LR_SCHEDULER = AttrDict()
config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 40
config.TRAIN.LR_SCHEDULER.NAME = 'cosine'
config.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
config.TRAIN.OPTIMIZER = AttrDict()
config.TRAIN.OPTIMIZER.NAME = 'adamw'
config.TRAIN.OPTIMIZER.EPS = 1e-8
config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
config.TRAIN.OPTIMIZER.MOMENTUM = 0.9

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

train_path = 'D:/kaggle/gan/data/preprocessing/train/photo_train_csv.csv'
train_monet = 'D:/kaggle/gan/data/preprocessing/train/monet_train_csv.csv'

test_path = 'D:/kaggle/gan/data/preprocessing/test/photo_test_csv.csv'
test_monet = 'D:/kaggle/gan/data/preprocessing/test/monet_test_csv.csv'

train_photo, test_photo = dataloader(train_path, test_path, 4)
train_monet, test_monet = dataloader(train_monet, test_monet, 4)

discriminator = resnet50(pretrained = True)
discriminator.fc = nn.Linear(2048, 1)
discriminator = discriminator.to('cuda')


criterion1 = nn.MSELoss()
criterion2 = nn.BCEWithLogitsLoss()
generator = SwinUnet()
generator.load_state_dict(torch.load('D:/kaggle/gan/model_weight/pretrained/swin_base_patch4_window7_224.pth'), \
                          strict = False)
generator.to('cuda')
optimizer_g = build_optimizer(config, generator)
optimizer_d = build_optimizer(config, discriminator)
lr_scheduler_g = build_scheduler(config, optimizer_g, len(train_photo) // config.TRAIN.ACCUMULATION_STEPS)
lr_scheduler_d = build_scheduler(config, optimizer_d, len(train_monet) // config.TRAIN.ACCUMULATION_STEPS)

save_path = 'D:/kaggle/gan/model_weight/train'

epochs = 100
def train(project):
    wandb.init(project="gan_monet", entity="sungkyunkwan_", name=project)
    min_gen_loss = 100
    min_dis_loss = 100
    min_total_loss = 100
    for epoch in tqdm(range(epochs)):
        discriminator_losses = []
        generator_losses = []
        for train_monet_idx, (train_mo) in tqdm(enumerate(train_monet)):
           for train_photo_idx, (train_img) in enumerate(train_photo):

                with torch.no_grad():
                    train_img = train_img.to('cuda')
                    B_p, C, H, W = train_img.shape
                    fake_img = generator(train_img)
                fake_label = torch.zeros([B_p, 1], dtype = torch.float32).to('cuda')

                real_img = train_mo.to('cuda')
                B_m, C, H, W = real_img.shape
                real_label = torch.ones([B_m, 1], dtype = torch.float32).to('cuda')

                fake_pred = discriminator(fake_img)
                fake_loss = criterion2(fake_pred, fake_label)

                real_pred = discriminator(real_img)
                real_loss = criterion2(real_pred, real_label)

                discriminator_loss = fake_loss + real_loss
                optimizer_d.zero_grad()
                discriminator_loss.backward()
                optimizer_d.step()
                discriminator_losses.append(discriminator_loss.detach().cpu().numpy())

                if ((train_photo_idx+1) % 5 == 0):
                    b_r,c_r,h_r,w_r = real_img.shape
                    b_f, c_f, h_f, w_f = fake_img.shape
                    if b_r == b_f :
                        fake_img = generator(train_img)
                        generator_loss = criterion1(fake_img, real_img)
                        optimizer_g.zero_grad()
                        generator_loss.backward()
                        optimizer_g.step()
                        generator_losses .append(generator_loss.detach().cpu().numpy())

                if ((train_photo_idx + 1) % 120 == 0 ):
                    print('#'* 50)
                    print('Generator Loss : {:.4f}'.format(generator_loss.detach().cpu().numpy()))
                    print('Discriminator Loss : {:.4f}'.format(discriminator_loss.detach().cpu().numpy()))
                    print('Total Loss : {:.4f}'.format(generator_loss.detach().cpu().numpy() + \
                                                       discriminator_loss.detach().cpu().numpy()))

        generator_losses = np.array(generator_losses)
        discriminator_losses = np.array(discriminator_losses)
        mean_g_loss = np.mean(generator_losses)
        mean_d_loss = np.mean(discriminator_losses)
        mean_total_loss = mean_g_loss + mean_d_loss
        wandb.log({'Test mean Discriminator': mean_d_loss})
        wandb.log({'Test mean Generator': mean_g_loss})
        wandb.log({'Test mean Total': mean_total_loss})

        generator_losses = []
        discriminator_losses = []
        with torch.no_grad():
            for test_monet_idx, (test_mo) in tqdm(enumerate(test_monet)):
                for test_photo_idx, (test_img) in enumerate(test_photo):

                    test_real_img = test_mo.to('cuda')

                    test_img = test_img.to('cuda')
                    test_fake_img = generator(test_img)


                    B_f, C, H, W = test_fake_img.shape
                    t_fake_label = torch.zeros([B_f, 1], dtype = torch.float32).to('cuda')
                    B_r, C, H, W = test_real_img.shape
                    t_real_label = torch.ones([B_r, 1], dtype = torch.float32).to('cuda')

                    fake_pred = discriminator(test_fake_img)
                    real_pred = discriminator(test_real_img)
                    d_fake_loss = criterion2(fake_pred, t_fake_label)
                    d_real_loss = criterion2(real_pred, t_real_label)

                    t_d_loss = d_fake_loss.detach().cpu().numpy() + d_real_loss.detach().cpu().numpy()
                    discriminator_losses.append(t_d_loss)
                    if B_f == B_r:
                        t_g_loss = criterion1(test_fake_img, test_real_img).detach().cpu().numpy()
                        generator_losses.append(t_g_loss)

        discriminator_losses = np.array(discriminator_losses)
        generator_losses = np.array(generator_losses)

        mean_d_loss = np.mean(discriminator_losses)
        mean_g_loss = np.mean(generator_losses)
        mean_total_loss = mean_g_loss + mean_d_loss

        print('Test Epoch Loss : {:.4f}'.format(mean_total_loss))
        print('Test Gen Loss : {:.4f}'.format(mean_g_loss))
        print('Test Dis Loss : {:.4f}'.format(mean_d_loss))
        wandb.log({'Test mean Total': mean_total_loss})
        wandb.log({'Test mean Genearator': mean_g_loss})
        wandb.log({'Test mean Discriminator': mean_d_loss})


        if mean_d_loss < min_dis_loss :
            min_dis_loss = mean_d_loss
            torch.save(discriminator.state_dict(), 'D:/kaggle/gan/model_weight/train/discriminator/Epoch_{}'.format(epoch+1))
            print('WoW New Discriminator Model : {:.4f}'.format(min_dis_loss))
        if mean_g_loss < min_gen_loss :
            min_gen_loss = mean_g_loss
            torch.save(generator.state_dict(), 'D:/kaggle/gan/model_weight/train/generator/Epoch_{}'.format(epoch+1))
            print('WoW New Generator Model : {:.4f}'.format(min_gen_loss))

        if mean_total_loss < min_total_loss :
            min_total_loss = mean_total_loss

            torch.save(discriminator.state_dict(), 'D:/kaggle/gan/model_weight/train/total/Epoch_{}'.format(epoch+1))
            torch.save(generator.state_dict(), 'D:/kaggle/gan/model_weight/train/total/Epoch_{}'.format(epoch+1))

            print('WoW New Total Model : {:.4f}'.format(min_total_loss))
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str)

    args = parser.parse_args()

    train(project=args.project)