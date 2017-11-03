#!/usr/bin/python2
# encoding=utf-8
'''
Created on 2017年10月11日

@author: kohillyang
'''
import sys
from showdataset import Ai_data_set
import mxnet as mx
import numpy as np

# sys.path.append("/data1/yks/mxnet_ai/mxnet_pose_for_AI_challenger")
from modelCPMWeight import CPMModel,numofparts,numoflinks
save_prefix  = "../outputs/models/yks_pose"
def getModule(prefix=None , begin_epoch=0, batch_size=10,re_init = False,gpus = [2,3],use_resnet = True):

    if re_init:
        print("reinit")
        sym = CPMModel(use_resnet = True)
        # batch_size = 2
        # mx.viz.plot_network(sym,shape = {"data":(1,3,368,368),        
        #             'heatmaplabel':(batch_size, numofparts, 46, 46),
        #             'partaffinityglabel':(batch_size, numoflinks * 2, 46, 46),
        #             'heatweight':(batch_size, numofparts, 46, 46),
        #             'vecweight':(batch_size, numoflinks * 2, 46, 46)
                
        # },title="plot",hide_weights=False).view()   
        sym_resnet,args_resnet,_ = mx.model.load_checkpoint("/data1/yks/models/resnet/resnet-152", 0)
        from train_config import resnet_keys
        newargs = {}
        aux_resnet = {}
        for key in resnet_keys:
            newargs[key] = args_resnet[key]
    else:
        _,newargs,aux_resnet = mx.model.load_checkpoint(prefix, begin_epoch)
        sym = CPMModel(use_resnet = False)
        pass
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in gpus],
                        label_names=['heatmaplabel',
                                'partaffinityglabel',
                                'heatweight',
                                'vecweight'])
    model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],
            label_shapes=[
                    ('heatmaplabel', (batch_size, numofparts, 46, 46)),
                    ('partaffinityglabel', (batch_size, numoflinks * 2, 46, 46)),
                    ('heatweight', (batch_size, numofparts, 46, 46)),
                    ('vecweight', (batch_size, numoflinks * 2, 46, 46))])
    model.init_params(arg_params=newargs, aux_params=aux_resnet, allow_missing=True,allow_extra = True)

    return model
def train(cmodel,train_data,begin_epoch,end_epoch,batch_size,save_prefix,single_train_count = 4):
    cmodel.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 4e-4 ), ))         
    for nbatch,data_batch in enumerate(train_data):
        current_batch = begin_epoch + nbatch 
        if current_batch >= end_epoch:
            print("info: finish training.")
            return
        if nbatch % 100 == 0:
            cmodel.save_checkpoint(save_prefix, current_batch)
            print ("save_checkpoint finished")
        if nbatch % 10 != 0:
            for _ in range(single_train_count):
                cmodel.forward(data_batch, is_train=True) 
                cmodel.backward()  
                cmodel.update()
        else:
            sumerror=0

            cmodel.forward(data_batch, is_train=True)       # compute predictions  
            prediction=cmodel.get_outputs()

            lossiter = prediction[1].asnumpy()              
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'start heat: ', cls_loss
                
            lossiter = prediction[0].asnumpy()
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'start paf: ', cls_loss
            
            lossiter = prediction[-1].asnumpy()              
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'end heat: ', cls_loss
            
            lossiter = prediction[-2].asnumpy()
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'end paf: ', cls_loss   
            print(current_batch,end_epoch,sumerror)

            sumerror=0

            for _ in range(single_train_count):
                cmodel.forward(data_batch, is_train=True) 
                cmodel.backward()  
                cmodel.update()

            cmodel.forward(data_batch, is_train=True)       # compute predictions  
            prediction=cmodel.get_outputs()

            lossiter = prediction[1].asnumpy()              
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'start heat: ', cls_loss
                
            lossiter = prediction[0].asnumpy()
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'start paf: ', cls_loss
            
            lossiter = prediction[-1].asnumpy()              
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'end heat: ', cls_loss
            
            lossiter = prediction[-2].asnumpy()
            cls_loss = np.sum(lossiter)/batch_size
            sumerror = sumerror + cls_loss
            print 'end paf: ', cls_loss   
            print(current_batch,end_epoch,sumerror)            
            print("*******************************")
                
if __name__ == "__main__":

    start_epoch = 0
    batch_size = 40
    cpm_model = getModule(save_prefix,start_epoch,batch_size,False,use_resnet = False)
    train_data = Ai_data_set(batch_size,"ai_inf_v1.db")
    train(cpm_model,train_data,start_epoch,99999,batch_size,save_prefix,1)





