import os
import torch
import numpy as np
import matplotlib.pyplot as plt

args = {'epochs': 300,
		  'save_dir': 'save_temp'}

def getCheckpoints(num_epochs):
	for epoch in range(num_epochs):
		fname = os.path.join(args['save_dir'], 'checkpoint_{}.tar'.format(epoch))
		if os.path.isfile(fname):
			print("=> loading checkpoint '{}'".format(fname))
			checkpoint = torch.load(fname)
			yield checkpoint
		else:
			break

def plotData():
	epochs = np.arange(args['epochs'])
	trainLosses, trainAccs, trainTimes = np.empty((0,)), np.empty((0,)), np.empty((0,))
	valLosses, valAccs = np.empty((0,)), np.empty((0,))
	bestAcc, bestEpoch = float('-inf'), None
	epoch = 0
	for checkpoint in getCheckpoints(args['epochs']):
		trainLosses = np.append(trainLosses, checkpoint['train_loss'], axis=0)
		trainAccs = np.append(trainAccs, checkpoint['train_acc'], axis=0)
		trainTimes = np.append(trainTimes, checkpoint['train_time'], axis=0)
		valLosses = np.append(valLosses, checkpoint['val_loss'], axis=0)
		valAccs = np.append(valAccs, checkpoint['val_acc'], axis=0)
		if checkpoint['best_prec1'] > bestAcc:
			bestAcc = checkpoint['best_prec1']
			bestEpoch = epoch
		epoch += 1
	plt.plot(epochs, trainLosses)
	plt.ylabel('Train Loss')
	plt.xlabel('Epoch')
	plt.title('Train Losses')
	plt.show()
	plt.plot(epochs, trainAccs)
	plt.ylabel('Train Accuracy')
	plt.xlabel('Epoch')
	plt.title('Train Accuracies')
	plt.show()
	plt.plot(epochs, trainTimes)
	plt.ylabel('Train Time')
	plt.xlabel('Epoch')
	plt.title('Train Times')
	plt.show()
	plt.plot(epochs, valLosses)
	plt.ylabel('Validation Loss')
	plt.xlabel('Epoch')
	plt.title('Validation Losses')
	plt.show()
	plt.plot(epochs, valAccs)
	plt.ylabel('Validation Accuracy')
	plt.xlabel('Epoch')
	plt.title('Validation Accuracies')
	plt.show()
	print('Best validation accuracy:', bestAcc, ', at epoch', bestEpoch)