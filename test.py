import scipy
# from srfm import *
# from srdfm import *
from self_utils import *
# from srdfm import *
import time
import os


import cv2 as cv
import scipy.io as sio
# from architecture.AWAN import *
# from architecture.Restormer import *
# from architecture.MIRNet import *
# from architecture.hinet import *
# from architecture.MPRNet import *
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import datetime
# from option import opt
from self_option import opt
import matplotlib.pyplot as plt

# plt.plot(range(le/////n(a)),a)
# from MIRNet import *
# from HINet import *
# from AWAN import *
# plt.show()
import scipy
import cv2
# from CNN3d import *
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '3'






if not torch.cuda.is_available():
    raise Exception('NO GPU!')



PATH6='/data1/lxx_data/MST2222/'
opt.outf='/data1/lxx_data/SR/fctft.pth'

per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch
print("\nloading dataset ...")

val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)




##################################3
def gene_vis3dcnn(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    model=CNN3d()
    x=x.unsqueeze(0)
    x=x.unsqueeze(1)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)




    data=data.squeeze(0)
    checkpoint = torch.load("/data1/lxx_data/CNN3d/3dcnn.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/cnn3d.mat", {"msi":A})

def gene_visMPR(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = MPRNet()
    checkpoint = torch.load("/data1/lxx_data/SR_model/mprnet.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/mprnet.mat", {"msi":A})

def gene_vismst(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = MST_Plus_Plus()
    checkpoint = torch.load("/data1/lxx_data/SR_model/mst_plus_plus.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/mst_pp.mat", {"msi":A})

def gene_visHinet(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = HINet()
    checkpoint = torch.load("/data1/lxx_data/SR_model/hinet.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/hinet.mat", {"msi":A})

from restoremer import *
def gene_visresotrmer(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = Restormer()
    checkpoint = torch.load("/data1/lxx_data/SR_model/restormer.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    ##################
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)
    data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/restormer.mat", {"msi":A})

def gene_vishscnn(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = HSCNN_Plus()
    checkpoint = torch.load("/data1/lxx_data/SR_model/hscnn_plus.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    ssim = torch_ssim(orig, data)
    print(ssim)
    psnr = torch_psnr(orig, data)
    print(psnr)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/hscnn_plus.mat", {"msi":A})


def gene_vishhdnet(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = HDNet()
    checkpoint = torch.load("/data1/lxx_data/SR_model/hdnet.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    ssim = torch_ssim(orig, data)
    print(ssim)
    psnr = torch_psnr(orig, data)
    print(psnr)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/hdnet.mat", {"msi":A})


def gene_visawan(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = AWAN()
    checkpoint = torch.load("/data1/lxx_data/SR_model/awan.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)

    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/awan.mat", {"msi":A})
from MIRNet import *
def gene_vismirnet(n):

    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1)
    checkpoint = torch.load("/data1/lxx_data/SR_model/mirnet.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)




    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/mirnet.mat", {"msi":A})

# from srdfm import  *


def gene_visfctft(n):
    x=val_data[n][0]
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    model = FCTFT()
    checkpoint = torch.load("/data1/lxx_data/SR_model/fctft.pth")
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
    data=model(x)
    orig = val_data[n][1]
    orig = torch.from_numpy(orig)
    orig = orig.unsqueeze(0)
    psnr = torch_psnr(orig, data)
    print(psnr)
    ssim = torch_ssim(orig, data)
    print(ssim)
    data=data.squeeze(0)
    # data=data.squeeze(0)
    A=data.detach().cpu().numpy()
    A=np.transpose(A,(1,2,0))
    scipy.io.savemat("data_zls/mst_pp.mat", {"msi":A})


def gene_orig():
    x=val_data[0][0]
    A=x
    scipy.io.savemat("data_zls/orig.mat", {"truth":A})

# gene_vis()
n=1
gene_visfctft(1)


