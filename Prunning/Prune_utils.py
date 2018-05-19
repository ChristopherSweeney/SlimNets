#Wrappers for prunning and training networks gradually
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import torchvision.models as models
import math
import torch.nn as nn
import torch.nn.init as init

#############################################################################
#utility functions
##############################################################################
"""
convinince functions adapted from https://github.com/ChristopherSweeney/pytorch-weights_pruning
"""
def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def prune_rate(model):
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc

def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    
    return acc


########################################################################################################
#layer-wise prunning wrapper class
#########################################################################################################


class PruneWrapSparse():
    def __init__(self,train_start,model,sparsity_initial,sparsity_target,prune_steps,update_rate,train):
        #model params
        self.model = model.float()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = train
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        
        #bookkeeping
        self.current_sparsity=0
        self.sparsity_target = sparsity_target
        self.sparsity_initial = sparsity_initial
        self.train_step = train_start
        self.prune_steps =prune_steps
        self.update_rate = update_rate
        self.prunning = True
        self.running_loss = 0
        self.loss_over_time=[]
        self.sparsity_over_time=[]
        self.epoch=[]
        self.train1 = 0.0
        
    def to_string(self):
        print "current sparsity: " + str(self.current_sparsity)
        print "sparsity target: "+ str(self.sparsity_target)
        print "initial sparsity: " + str(self.sparsity_initial)
        print "current train step: "+ str(self.train_step)
        print "prune steps: "+str(self.prune_steps)
        print "prunning rate: "+str(self.update_rate)
        
    def test(self,test_data_loader):
        self.model.eval()
        
        correct, samples = 0, len(test_data_loader.dataset)
        for x, y in test_data_loader:
            x_var = to_var(x, volatile=True)
            scores = self.model(x_var)
            _, preds = scores.data.cpu().max(1)
            correct += (preds == y).sum()

        return float(correct)/samples
    
    def reset(self):
        self.current_sparsity=0.
        self.train_step = 0.
        self.current_sparsity = 0.
    
    def compute_current_target(self):
        return self.sparsity_target+(self.sparsity_initial-self.sparsity_target)*(1-(float(self.train_step)/(self.prune_steps*self.update_rate)))**3
    
    def train(self, optimizer = None, epoches = 10):
        print "training"
        start = time.time()
        for i in range(epoches):
            print self.to_string()
            print "epoch: ", i
            print self.current_sparsity
            for batch, label in self.train_loader:
                #stop prunning?
                if self.prunning and self.current_sparsity >= self.sparsity_target:
                    self.prunning = False
                    print "finished prunning"
                #should i prune?
                if self.train_step%self.update_rate==0:
                    print self.train_step
                    print self.running_loss/float(self.update_rate)
                    if self.prunning:
                        self.prune()
                    self.running_loss = 0.0
                    print prune_rate(self.model)

                #train
                batch,label = (to_var(batch),to_var(label))
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch), label)
                loss.backward()
                self.optimizer.step()
                self.running_loss+=loss.data[0]
                self.train_step+=1
                self.train1+=loss.data[0]
                if self.train_step %3000 ==0:
                    self.epoch.append(self.test(testloader))
                    self.train1 = 0
            print prune_rate(self.model)
            print self.test(testloader)
        print "Finished fine tuning."
        print "Time elapsed:" + str((time.time() - start))

    def prune(self):#all conv layers for now
        current_sparsity_target = self.compute_current_target()
        print current_sparsity_target
        print self.to_string()
        for seq in list(self.model.children()):
            for layer in seq:
                if isinstance(layer, MaskedLinear) or isinstance(layer, MaskedConv2d):
                    #find weight threshold for layer
                    weight_threshold = self.get_weight_threshold(layer.parameters(),current_sparsity_target) 
                    #set mask
                    mask = self.get_prune_mask(layer,weight_threshold)
                    #prune
                    layer.set_mask(mask)
        self.sparsity_over_time.append(current_sparsity_target)
        self.current_sparsity = current_sparsity_target # more accurate actually figure out hw much was prunned
        self.loss_over_time.append(self.running_loss/float(self.update_rate))
    
        
    def get_prune_mask(self,layer,threshold):
        # generate mask
        p=layer.weight.data.abs()
        pruned_inds = p > threshold
        return pruned_inds.float()
        
    def save_weight_hist(self,params):
        params = []
        plt.figure()
        for p in self.model.parameters():
            params.extend(list(p.data.cpu().abs().numpy().flatten()))
        plt.hist(params,bins=100,range=[0, 1])
        plt.show()
        
    def get_weight_threshold(self,params,percent_to_prune):
        weights=[]
        for param in params:
            if len(list(param.cpu().data.abs().numpy().flatten()))>1:
                weights.extend(list(param.cpu().data.abs().numpy().flatten()))
        threshold = np.percentile(np.array(weights),percent_to_prune*100)
        #print len(np.nonzero(np.array(weights)))/float(len(weights))
        return threshold

    def calculate_current_model_size(self):
        total_nb_param = 0
        non_zero_param = 0
        for parameter in self.model.parameters():
            total_nb_param += np.size(parameter.cpu().data.abs().numpy())
            non_zero_param += np.count_nonzero(parameter.cpu().data.abs().numpy()>0)
        print "model sparsity: ", 1-float(non_zero_param)/total_nb_param
        bits = 32 #32 bit floats
        return bits*non_zero_param
   
    def calculate_current_model_inference_time(self,samples=10):
        time_sum =0
        for i in range(samples):
            (batch, label) = next(iter(self.train_loader))
            input = to_var(batch.float())
            start = time.time()
            output = self.model(input)
            time_sum+=time.time() - start
        print "Time elapsed:" + str((time_sum/float(samples)))    
            