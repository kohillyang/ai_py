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
sys.path.append("/data1/yks/mxnet_ai/mxnet_pose_for_AI_challenger")
from modelCPMWeight import CPMModel
numofparts = 15
numoflinks = 13
save_prefix  = "../outputs/models/yks_pose"
def getModule(prefix=None , begin_epoch=0, batch_size=10,re_init = False,gpus = [1]):
    if re_init:
        from train_config import vggparams,params_realtimePose_layers
        vgg19_prefix = "/data1/yks/models/vgg19/vgg19"
        mpi_prefix = "/data1/yks/models/openpose/realtimePose_mpi"
        # coco_prefix = "/data1/yks/models/openpose/realtimePose"
        sym_vgg, arg_vgg, aux_vgg = mx.model.load_checkpoint(vgg19_prefix, 0)
        sym_mpi, arg_mpi, aux_mpi = mx.model.load_checkpoint(mpi_prefix, 0)
        newargs = {}
        for key in params_realtimePose_layers:
            key_weight = key + "_weight"
            key_bias = key + "_bias"
            newargs[key_weight] = arg_mpi[key_weight]
            newargs[key_bias] = arg_mpi[key_bias]

        # for key in vggparams:
        #     newargs[key] = arg_vgg[key]
        sym = CPMModel()
    else:
        sym, newargs, _ = mx.model.load_checkpoint(prefix, begin_epoch)
    mx.viz.plot_network(sym).view()
        
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
    model.init_params(arg_params=newargs, aux_params={}, allow_missing=True)

    return model
def train(cmodel,train_data,begin_epoch,end_epoch,batch_size,save_prefix,single_train_count = 4):
    cmodel.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.000004 ), ))         
    for nbatch,data_batch in enumerate(train_data):
        current_batch = begin_epoch + nbatch 
        if current_batch >= end_epoch:
            print("info: finish training.")
            return
        if nbatch % 50 == 0:
            cmodel.save_checkpoint(save_prefix, current_batch)
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
    batch_size = 1
    cpm_model = getModule(save_prefix,start_epoch,batch_size,True)
    train_data = Ai_data_set(batch_size)
    train(cpm_model,train_data,start_epoch,9999,batch_size,save_prefix,4)





