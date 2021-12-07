from torch._C import device
from unet_a2c import *
import cv2
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision
import torch.nn.functional as F

TRAIN_PATH = './echocardiography/train/A2C/'
VALIDATION_PATH='./echocardiography/validation/A2C/'
BATCH_SIZE = 4
NUM_EPOCHS = 10000
LEARNING_RATE = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W_SIZE = 600
H_SIZE = 400

# Train Dataset
file_list = os.listdir(TRAIN_PATH)
npy_list = list()
png_list = list()

for name in file_list:
    if name[-4:]=='.png':
        png_list.append(name)
    elif name[-4:]=='.npy':
        npy_list.append(name)

dataset = list()
for name in png_list:
    common_name = name[:-4]
    npy_obj = np.load(TRAIN_PATH+common_name+'.npy')
    npy_obj = cv2.resize(npy_obj, (W_SIZE,H_SIZE)).reshape(1,H_SIZE,W_SIZE)*255
    png_obj = cv2.imread(TRAIN_PATH+common_name+'.png', 0)
    png_obj = cv2.resize(png_obj, (W_SIZE,H_SIZE)).reshape(1,H_SIZE,W_SIZE)
    dataset.append((png_obj, npy_obj))

# Validation Dataset
file_list_val = os.listdir(VALIDATION_PATH)
npy_list_val = list()
png_list_val = list()

for name in file_list_val:
    if name[-4:]=='.png':
        png_list_val.append(name)
    elif name[-4:]=='.npy':
        npy_list_val.append(name)

dataset_val = list()
for name in png_list_val:
    common_name = name[:-4]
    npy_obj = np.load(VALIDATION_PATH+common_name+'.npy')
    npy_obj = cv2.resize(npy_obj, (W_SIZE,H_SIZE)).reshape(1,H_SIZE,W_SIZE)*255
    png_obj = cv2.imread(VALIDATION_PATH+common_name+'.png', 0)
    png_obj = cv2.resize(png_obj, (W_SIZE,H_SIZE)).reshape(1,H_SIZE,W_SIZE)
    dataset_val.append((png_obj, npy_obj))


unet = DoubleUNet(1,1).to(DEVICE)

train_dataset = dataset
test_dataset = dataset_val
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

BEST_MODEL_PATH = 'best_model_A2C.pth'

d_time = datetime.datetime.now()
folder_name = "runs_a2c/"+d_time.strftime("%Y%m%d%H%M%S")
os.mkdir(folder_name)
writer = SummaryWriter(log_dir=folder_name)
pre_test_loss = None
optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), weight_decay=0.0005)

for epoch in range(NUM_EPOCHS):
    train_loss=0.
    test_loss=0.
    for img, label in train_loader:
        img = img.to(DEVICE)
        label = label.float().to(DEVICE)
        b_size = img.shape[0]
        
        optimizer.zero_grad()
        outputs_, adap_outputs, adap_label = unet(img.float(), label)
        outputs = outputs_
        loss1 = torch.sum(torch.square((outputs-label)))
        loss2 = torch.sum(torch.square((adap_outputs-adap_label)))
        loss = torch.sqrt(loss1+loss2)
        train_loss+=loss        
        loss.backward()
        optimizer.step()
       
    outputs_test=None    
    label_test=None   
    img = None
    TP_collect = list()
    LABEL_TRUE_SUM_collect = list()
    PREDICTI_TRUE_SUM_collect = list()
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(DEVICE)
            label_test = label.float().to(DEVICE)
                        
            outputs_test,  adap_outputs, adap_label = unet(img.float(), label_test)
            loss1 = torch.sum(torch.square((outputs_test-label_test)))
            loss2 = torch.sum(torch.square((adap_outputs-adap_label)))

            loss = loss1+loss2
            test_loss+=loss
    
            # DSC = 2*TP/(2TP+FP+FN)
            # JI = DICE/(2-DICE)       
            outputs_test_for_grid = torch.round(unet.get_output())
            cal_outputs_test_for_grid = outputs_test_for_grid.int()

            cal_label_test = label_test.int()
            cal_label_test = torch.round(cal_label_test/255.).int()
            TP_collect.append(torch.sum(cal_outputs_test_for_grid & cal_label_test))
            LABEL_TRUE_SUM_collect.append(torch.sum(cal_label_test))
            PREDICTI_TRUE_SUM_collect.append(torch.sum(cal_outputs_test_for_grid))

    TP = np.sum(TP_collect)
    LABEL_TRUE_SUM = np.sum(LABEL_TRUE_SUM_collect)
    PREDICTI_TRUE_SUM = np.sum(PREDICTI_TRUE_SUM_collect)


    DSC = 2*TP/(LABEL_TRUE_SUM+PREDICTI_TRUE_SUM)
    JI = DSC/(2-DSC)
    pre_test_loss = DSC

    writer.add_scalar("SUMMARY_A2C/DSC",DSC,epoch)
    writer.add_scalar("SUMMARY_A2C/JI LOSS",JI,epoch)
    grid_test = torchvision.utils.make_grid(outputs_test_for_grid)
    writer.add_image('PREDICTION_A2C', grid_test, epoch)
    grid_label = torchvision.utils.make_grid(label)
    writer.add_image('LABEL_A2C', grid_label, epoch)
    grid_img = torchvision.utils.make_grid(img)
    writer.add_image('IMAGE_A2C', grid_img, epoch)
    if pre_test_loss is None:
        pre_test_loss = test_loss
    if pre_test_loss<test_loss:
        print(str(epoch),' Best A2C DSC:',DSC.item(), 'JI: ', JI.item())
        pre_test_loss= test_loss
        torch.save(unet.state_dict(), str(epoch)+BEST_MODEL_PATH)

