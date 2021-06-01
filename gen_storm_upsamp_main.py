import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from dataAndOperators import dataAndOperators
from generator_upsamp import generator
from optimize_generator import optimize_generator
from latentVariable import latentVariable
from ptflops import get_model_complexity_info

gpu=torch.device('cuda:0')

params = {'name':'parameters',
     'directory':'',
     'device':gpu,
     'filename':"",
     'dtype':torch.float,     
     
     'im_size':(340,340),   #imaege size
     'nintlPerFrame':6,    # interleaves per frame
     'nFramesDesired':150,  # number of frames in the reconstruction
     'slice':3,            # slice of the series to process
     'factor':1,           # scale image by 1/factor to save compute time
     
     'gen_base_size': 42,   # base number of filters
     'gen_reg': 0.0005,       # regularization penalty on generator

     'siz_l':2} # number of latent parameters 

params['directory'] = '/Users/zouqing/paper_exp/'
params['filename']  = '/Users/zouqing/paper_exp/Series8.mat'
 
#%% Level 1 training

nFramesDesired = 150
nScale = 50
params['nintlPerFrame'] = int(6*nScale)
params['nFramesDesired'] = int(nFramesDesired/nScale)
params['lr_g'] = 1e-4
params['lr_z'] = 0e-4

# Reading and pre-processing the data and parameters
dop = dataAndOperators(params)

# Initializaition of the generator
G = generator(params)
G.weight_init()
G.cuda(gpu)

# Initialization of th elatent variables
alpha = [0,0]
alpha = torch.FloatTensor(alpha).to(gpu)
z = latentVariable(params,init='ones',alpha=alpha)
# Training
G,z,train_hist,SER1 = optimize_generator(dop,G,z,params,train_epoch=2000,proj_flag=False)


#%% Level 2 training

G_old = G.state_dict()
z_old = z.z_

nFramesDesired = 150
nScale = 3

params['nintlPerFrame'] = int(6*nScale)
params['nFramesDesired'] = int(nFramesDesired/nScale)
params['lr_g'] = 1e-4
params['lr_z'] = 3e-3


# Change the number of frames
dop.changeNumFrames(params)

#Initialize the latent variables
alpha = [2,0.5]
alpha = torch.FloatTensor(alpha).to(gpu)

z.z_ = z_old
z = latentVariable(params,z_in=z,alpha=alpha)
G.load_state_dict(G_old)
#Training
G,z1,train_hist,SER2 = optimize_generator(dop,G,z,params,train_epoch=800,proj_flag=True)

#%% Final training

G_old = G.state_dict()
z_old = z1.z_
params['nFramesDesired'] = nFramesDesired
params['nintlPerFrame'] = 6
params['lr_g'] = 7e-5
params['lr_z'] = 1e-3

dop.changeNumFrames(params)

alpha = [2,0.5]
alpha = torch.FloatTensor(alpha).to(gpu)

z.z_ = z_old
z = latentVariable(params,z_in=z,alpha=alpha)
G.load_state_dict(G_old)

G,z,train_hist,SER3 = optimize_generator(dop,G,z,params,train_epoch=700,proj_flag=True)

#%% Display the results
ztemp = z.z_.data.detach()


#%% Save the results

zs = z.z_.data.squeeze().cpu().numpy()
torch.save(G.state_dict(), "generator_param.pkl")
np.save('zs.npy', zs)

#%% Saving data to file
import imageio
from matplotlib.transforms import Bbox
import  os

pad = nn.ReplicationPad2d((1,0,1,0))
blur = nn.MaxPool2d(2, stride=1)

images = []
my_dpi = 100 # Good default - doesn't really matter
h = params['im_size'][0]
w = params['im_size'][1]

dirname =  params['filename'].replace('.mat','/results')
dirname = dirname+str(params['slice'])
if not(os.path.exists(dirname)):
    os.makedirs(dirname)
    
for k in range(nFramesDesired):
    test_image = ztemp[k,:,:,:].unsqueeze(0)
    test_image = blur(pad(G(test_image)))
    test_image1 = test_image.squeeze().data.cpu().numpy()
    test_image1 = test_image1[0,:,:] + test_image1[1,:,:]*1j
    fig, ax = plt.subplots(1, figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    ax.set_position([0,0,1,1])

    plt.imshow((abs(test_image1)), cmap='gray')
    ax.axis('off')
    img_name = dirname+'/frame_' + str(k) + '.png'
    fig.savefig(img_name,bbox_inches=Bbox([[0,0],[w/my_dpi,h/my_dpi]]),dpi=my_dpi)
    plt.close()
    images.append(imageio.imread(img_name))
imageio.mimsave(dirname+'.gif', images, fps=20)
