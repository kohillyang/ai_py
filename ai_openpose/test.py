#!/usr/bin/python2
# encoding=utf-8
import sys,cv2,os,argparse,copy
sys.path.append("/data1/yks/mxnet_ai/mxnet_pose_for_AI_challenger")
from modelCPMWeight import CPMModel_test
from train import numoflinks,numofparts,save_prefix

import mxnet as mx
import numpy as np
max_img_shape = (368,368)
imgshape_bind = (max_img_shape,)
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
def getHeatAndPAF(img_path,models):
    oriImg = cv2.imread(img_path) # B,G,R order
    oriImg = padimg(oriImg,max_img_shape[0])
    class DataBatch(object):
        def __init__(self, data, label, pad=0):
            self.data = [data]
            self.label = 0
            self.pad = pad
    def preprocess(x_):
        r = []
        for size in imgshape_bind:
            imgs_resize = cv2.resize(x_, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
            imgs_transpose = np.transpose(np.float32(imgs_resize[:,:,:]), (2,0,1))/256 - 0.5
            imgs_batch = DataBatch(mx.nd.array([imgs_transpose[:,:,:]]), 0)
            r.append(imgs_batch)
        return r
    def suffix_heatmap(heatmap,size):
        heatmap = np.moveaxis(heatmap.asnumpy()[0], 0, -1)
        heatmap = cv2.resize(heatmap, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)            
        return heatmap
    def suffix_paf(paf,size):
        paf = np.moveaxis(paf.asnumpy()[0], 0, -1)
        paf = cv2.resize(paf, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
        return paf
    
    def _getHeatPAF(args):
        onedata,model = args
        model.forward(onedata)
        result = model.get_outputs()
        max_shape = (oriImg.shape[0],oriImg.shape[1])
        heatmap = suffix_heatmap(result[1],max_shape)
        paf =     suffix_paf(    result[0],max_shape)
        return (heatmap,paf)

    imgs = preprocess(oriImg)
    results = _getHeatPAF((imgs[0],models[0]))
    heatmap_avg,paf_avg = results

    return img_path,oriImg,heatmap_avg,paf_avg
def getModel(prefix,epoch,gpus = [0]):
    sym  = CPMModel_test()
    batch_size = 1
    _, newargs, _ = mx.model.load_checkpoint(prefix, epoch)
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in gpus],                        
                          label_names=['heatmaplabel',
                                'partaffinityglabel',
                                'heatweight',
                                'vecweight'])
    model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],for_training = False)
    model.init_params(arg_params=newargs, aux_params={}, allow_missing=False)
    return model
def test(imgs_path):
    cmodel = getModel(save_prefix,2600)
    for x,y,z in os.walk(imgs_path):
        for name in z:
            path = os.path.join(x,name)
            img_path,oriImg,heatmap_avg,paf_avg = getHeatAndPAF(path,[cmodel])


            heatmap_avg[:,:,0] /= np.max(heatmap_avg[:,:,14])
            cv2.imshow("heatmap_avg",np.float32( heatmap_avg[:,:,14]))
            cv2.imshow("oriImg",oriImg)
            
            for i in range(15):
                heat_c = oriImg.astype(np.float32) /255
                heat_c[:,:,2] = 1 * heat_c[:,:,2] + heatmap_avg[:,:,i] * 255     
                heat_c[:,:,2] /= np.max(heat_c[:,:,2])
                cv2.imshow("heatmap_avg",heat_c)
                cv2.imshow("paf_avg",paf_avg[:,:,1].astype(np.float32) * 128)
                cv2.imshow("oriImg",oriImg)
            i = 12
            pagmap_x = cv2.resize(paf_avg[:,:,i * 2],(oriImg.shape[0],oriImg.shape[1]),cv2.INTER_CUBIC)
            pagmap_y = cv2.resize(paf_avg[:,:,i * 2 + 1],(oriImg.shape[0],oriImg.shape[1]),cv2.INTER_CUBIC)
            # pag_c = copy.copy(oriImg)
            pag_c = np.sqrt(pagmap_x * pagmap_x + pagmap_y * pagmap_y )     * 255      
            pag_c /= np.max(pag_c)
            cv2.imshow("pag",pag_c)
            if cv2.waitKey(0) == 27:
                sys.exit(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir","-i", help="input images dir ",type = str,default = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911")
    args = parser.parse_args()   
    test(args.images_dir)
            
            
            
            
            
            
            
