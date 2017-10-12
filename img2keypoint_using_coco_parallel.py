#encoding=utf-8
'''
@Author: kohill
'''

## Include mxnet path: you should include your mxnet local path, if mxnet path is global, 
## you don't need to include it.
import sys
sys.path.append("./mxnet_Realtime_Multi-Person_Pose_Estimation")
sys.path.append("../models/openpose")
sys.path.append("../")
import cv2 as cv
import cv2 
import numpy as np
import scipy
import PIL.Image
import math
import time
import matplotlib
import gnumpy as gnp
debug = False
# %matplotlib inline
import pylab as plt
from generateLabelCPM import *
from modelCPM import *
#sym = mxnetModule()


search_ratio = [0.5,1,1.5,2]
imgshape_bind = [(int(368*x),int(368*x)) for x in search_ratio]
max_img_shape = (max(search_ratio)*368,max(search_ratio)*368)
import json
def padimg(img,destsize):
    s = img.shape
    print img.shape,destsize/s[1],destsize/s[1]

    if(s[0] > s[1]):
        img_d = cv2.resize(img,dsize = None,fx = 1.0 * destsize/s[0], fy = 1.0 * destsize/s[0])
        img_temp = np.ones(shape = (destsize,destsize,3),dtype=np.uint8) * 128
        sd = img_d.shape
        img_temp[0:sd[0],0:sd[1],0:sd[2]]=img_d
    else:
        img_d = cv2.resize(img,dsize = None,fx = 1.0 * destsize/s[1],fy = 1.0 * destsize/s[1])
        img_temp = np.ones(shape = (destsize,destsize,3),dtype=np.uint8) * 128 
        sd = img_d.shape
        img_temp[0:sd[0],0:sd[1],0:sd[2]]=img_d
    return img_temp
def getModel(imgshape_for_bind,gpus = [1,2,5,7]):
    output_prefix='../models/openpose/realtimePose'
    sym, arg_params, aux_params = mx.model.load_checkpoint(output_prefix, 0)
    cmodels = []
    for i in range(len(search_ratio)):
        ctx = mx.gpu(gpus[i])
        cmodel0 = mx.mod.Module(symbol=sym, label_names=[],context = ctx)
        cmodel0.bind(data_shapes=[('data', (1, 3, imgshape_for_bind[i][0], imgshape_for_bind[i][1]))],for_training=False)
        cmodel0.init_params(arg_params=arg_params, aux_params=aux_params)
        cmodels.append(cmodel0)
    return cmodels
class HeatPafCalculator(object):
    class DataBatch(object):
        def __init__(self, data, label, pad=0):
            self.data = [data]
            self.label = 0
            self.pad = pad
    def preprocess(self,x_):
        r = []
        for size in imgshape_bind:
            imgs_resize = cv2.resize(x_, (size[0], size[1]), interpolation=cv.INTER_CUBIC)
            imgs_transpose = np.transpose(np.float32(imgs_resize[:,:,:]), (2,0,1))/256 - 0.5
            imgs_batch = HeatPafCalculator.DataBatch(mx.nd.array([imgs_transpose[:,:,:]]), 0)
            r.append(imgs_batch)
        return r
    def suffix_heatmap(self,heatmap,size):
        print(heatmap.shape)
        heatmap = np.moveaxis(heatmap.asnumpy()[0], 0, -1)
        print(heatmap.shape)
        heatmap = cv.resize(heatmap, (size[0], size[1]), interpolation=cv.INTER_CUBIC)            
        return heatmap
    def suffix_paf(self,paf,size):
        paf = np.moveaxis(paf.asnumpy()[0], 0, -1)
        paf = cv.resize(paf, (size[0], size[1]), interpolation=cv.INTER_CUBIC)
        return paf
    def __init__(self,models):
        self.models = models
    def begin_calc(self,img_path):
        self.img_path = img_path
        oriImg = cv2.imread(img_path) # B,G,R order
        self.oriImg = padimg(oriImg,max_img_shape[0])
        self.imgs = self.preprocess(self.oriImg)

        for i in range(len(self.imgs)):
            for img,model in zip(self.imgs,self.models):
                model.forward(img)

    def get_results(self):
        oriImg = self.oriImg
        heatmap_avg = None
        paf_avg = None
        for model in self.models:
            result = model.get_outputs()
            max_shape = (oriImg.shape[0],oriImg.shape[1])
            heatmap = self.suffix_heatmap(result[1],max_shape)
            paf =     self.suffix_paf(    result[0],max_shape)     
            if heatmap_avg is None or paf_avg is None:
                paf_avg = gnp.garray(paf) 
                heatmap_avg = gnp.garray(heatmap)
            else:
                paf_avg += gnp.garray(paf) 
                heatmap_avg += gnp.garray(heatmap)
        heatmap_avg /= len(self.imgs)
        paf_avg /= len(self.imgs)

        return self.img_path,self.oriImg,heatmap_avg.as_numpy_array(),paf_avg.as_numpy_array()
 
if __name__ == "__main__":
    import multiprocessing,argparse
    from img2keypoint_using_coco import parse_heatpaf

    processes_num = 1
    calculator_batch = 1

    gpus_list =[
        [1,2,5,7],
        [7,5,2,1],
    ] * calculator_batch
    
    pool = multiprocessing.Pool(processes = processes_num)

    testpath = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911"
    testjsonpath = "/tmp/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir","-i", help="input images dir ",type = str,default = testpath)
    parser.add_argument("--json_dir","-o", help="output json dir ",type = str,default = testjsonpath)
    args = parser.parse_args()
    calculators =[]
    for i in range(calculator_batch):
        models = getModel(imgshape_bind,gpus_list[i])
        calculators.append(HeatPafCalculator(models))
    img_pathes = []
    for x,y,z in os.walk(args.images_dir):
        for name in z:
            img_pathes.append(os.path.join(x,name))
    assert len(img_pathes) % calculator_batch == 0
    steps = int(len(img_pathes)/calculator_batch)
#     steps = 20
    for index in range(steps):
        for i in range(calculator_batch):
            calculators[i].begin_calc(img_pathes[index * calculator_batch + i])
        for calculator in calculators:
            img_path,oriImg,heatma_av,pafs_av  = calculator.get_results()
            print(oriImg.shape,heatma_av.shape,pafs_av.shape)
            pool.apply_async(parse_heatpaf, (img_path,oriImg,heatma_av,pafs_av,args.json_dir,False))
    pool.close()
    pool.join()
