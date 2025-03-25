import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import madgrad
import torch
import numpy as np
from torchvision import transforms
import sys  
sys.path.insert(0, './myCode')
from RADAM import RAdam

import logging
import warnings

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class PLModel(pl.LightningModule):
    def __init__(self, name, model, loss = None, num_epochs = 0, path = None, lr = 0.001, wd = 0.0, optimName = 'MADGRAD'):
        super(PLModel, self).__init__()

        self.maxTestAcc = 0
        self.optimName = optimName
        self.maxValAcc = 0
        self.testAcc = 0
        self.lr = lr
        self.bestModel = None
        self.wd = wd
        self.name = name
        self.epoch = num_epochs
        self.model = model
        self.labelName = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

        if loss is None:
            print('No loss specified, using default')
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

        if path is None:
            self.PATH = 'supervised/'+self.optimName+self.name+'.ckpt'

        self.writer = self.optimName+ self.name
        
        
        
    def forward(self, x):    
        x = self.model(x)
            
        return x
    
    def configure_optimizers(self):
        if self.optimName == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimName == 'MADGRAD':
            optimizer = madgrad.MADGRAD(self.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.wd, eps = 1e-06)
        else:
            optimizer = RAdam(self.parameters(), lr=self.lr , weight_decay= self.wd)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-7, patience=10)
        sched1 = {'scheduler': scheduler, 'monitor': 'valLoss'}
        #print(optimizer)
        return {
           'optimizer': optimizer,
           'lr_scheduler': sched1
       }
    
    def training_step(self, batch, batch_idx):
        acc = 0
        x, y = batch
        
        #Siamo all'interno della classe quindi per rifermi alla rete utilizzo self (al posto di CNN(x))
        output = self(x)
        
        J = self.loss(output, y)
        
        #print(y)
        
        for i in range(x.shape[0]):
            #print(y[i], output[i])
            if y[i] == torch.argmax(output[i]):
                acc += 1
        
        pbar = {'train_acc' : acc/x.shape[0]}
        
        return {'loss' : J,
                'progress_bar': pbar}
    
    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['progress_bar']['train_acc'] for x in train_step_outputs]).mean()
        
        self.log('trainAcc', avg_train_acc, prog_bar=False)
        self.log('trainLoss', avg_train_loss, prog_bar=False)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        acc = 0
        x, y = batch
        n_class_correct = torch.zeros(len(self.labelName)).to('cuda')
        n_class_samples = torch.zeros(len(self.labelName)).to('cuda')

        output = self(x)
        
        J = self.loss(output, y)
        
        for i in range(x.shape[0]):
            label = torch.argmax(output[i])
            if y[i] == label:
                acc += 1
                n_class_correct[label] += 1
            n_class_samples[y[i]] += 1

        pbar = {'val_acc' : acc/x.shape[0]}

        for i in range(len(self.labelName)):
           pbar[self.labelName[i]] = n_class_correct[i]
           pbar[self.labelName[i] + 'TOT'] = n_class_samples[i]

        return {'loss' : J,
                'progress_bar': pbar}

    @torch.no_grad()
    def validation_epoch_end(self, val_step_outputs):
        n_class_correct = torch.zeros(len(self.labelName)).to('cuda')
        n_class_samples = torch.zeros(len(self.labelName)).to('cuda')
        bal_acc = 0.

        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()

        for i in range(len(self.labelName)):
           n_class_correct[i] = torch.tensor([x['progress_bar'][self.labelName[i]] for x in val_step_outputs]).sum()
           n_class_samples[i] = torch.tensor([x['progress_bar'][self.labelName[i] + 'TOT'] for x in val_step_outputs]).sum()

        for i in range(len(self.labelName)):
            if n_class_samples[i] != 0:
                bal_acc += n_class_correct[i]/n_class_samples[i]
                self.log('valAcc'+self.labelName[i], n_class_correct[i]/n_class_samples[i], prog_bar=False)

        bal_acc = bal_acc/len(self.labelName)

        self.log('valAcc', avg_val_acc, prog_bar=False)
        self.log('valLoss', avg_val_loss, prog_bar=True)
        self.log('balValAcc', bal_acc, prog_bar=True)
        
        
        if bal_acc > self.maxValAcc:
            self.maxValAcc = avg_val_acc
            self.bestModel = self.model.state_dict()
                
    
    def recoverBestModel(self):
        print("Recovering best model")
        self.model.load_state_dict(self.bestModel)

