import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

import numpy as np

args = {'K_vals': [5, 24, 48, 48, 64, 128, 128, 160, 192, 256, 320, 320, 320, 320, 320, 320],
		'save_dir': 'save_temp',
		'pretrained_path': 'model_best_cpu.tar'}

def vh_decompose(conv, K):
	'''
	  Create low-rank decomposition of the convolutional layer from original model
	'''
	v = nn.Conv2d(conv.in_channels, K, (conv.kernel_size[0], 1), stride=(conv.stride[0], 1), padding=(conv.padding[0], 0))
	h = nn.Conv2d(K, conv.out_channels, (1, conv.kernel_size[1]), stride=(1, conv.stride[1]), padding=(0, conv.padding[1]))
	return v, h

def lowrankify(orig_model, Ks):
	'''
	  Make rank constrained model by decomposing original convolutional layers
	'''
	new_layers = []
	for layer in orig_model.children():
		if isinstance(layer, nn.Conv2d):
			v, h = vh_decompose(layer, Ks.pop(0))
			new_layers.extend([v, h])
		else:
			new_layers.append(layer)
	return nn.Sequential(*new_layers)

def approx_lowrank_weights(orig_model, lowrank_model):
	o = iter(orig_model.modules())
	l = iter(lowrank_model.modules())
	next(o)
	next(l)
	for layer in o:
		v = next(l)
		if isinstance(layer, nn.Conv2d):
			h = next(l)
			next(l)
			w, b = layer.weight.detach().numpy(), layer.bias
			v.bias = torch.nn.Parameter(torch.zeros(v.out_channels))
			h.bias = torch.nn.Parameter(b.data.clone())
			
			N, C, D, D = w.shape
			w_t = w.transpose(1, 2, 3, 0)
			newW = np.zeros((C*D, D*N))
			for i1, i2, i3, i4 in np.ndindex(C, D, D, N):
				j1, j2 = (i1 - 1) * D + i2, (i4 - 1) * D + i3
				newW[j1, j2] = w_t[i1, i2, i3, i4]
			U, S, V = np.linalg.svd(newW)
			V = V.transpose()
			v_w = np.zeros((v.out_channels, v.in_channels) + v.kernel_size)
			h_w = np.zeros((h.out_channels, h.in_channels) + h.kernel_size)
			for k, c, j, t in np.ndindex(v_w.shape):
				v_w[k,c,j,t] = U[(c-1)*D + j, k] * np.sqrt(S[k])
			for n, k, t, j in np.ndindex(h_w.shape):
				h_w[n, k, t, j] = V[(n-1)*D + j, k] * np.sqrt(S[k])
			v.weight.data = torch.from_numpy(v_w).float()
			h.weight.data = torch.from_numpy(h_w).float()

def create_lowrank_model(orig_model):
    lrm = vgg.vgg19_bn()
    lrm.features = lowrankify(lrm.features, args['K_vals'])
    approx_lowrank_weights(orig_model.features, lrm.features)
    return lrm

def save_lowrank_model(lrm):
    torch.save({'model': lrm, 'epoch': 0}, os.path.join(args['save_dir'], 'lowrank-model.tar'))

if __name__ == '__main__':
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    print("Loading pretrained model...")
    pretrainedModel = torch.load(args['pretrained_path'])
    print("Creating low-rank model...")
    lrm = create_lowrank_model(pretrainedModel)
    print("Saving low-rank model...")
    save_lowrank_model(lrm)