#encoding=utf-8
'''
@Author: kohill
'''

## Include mxnet path: you should include your mxnet local path, if mxnet path is global, 
## you don't need to include it.
import sys,os,cv2,scipy,math,matplotlib
import cv2 as cv
import cv2 
import numpy as np
import scipy
import PIL.Image
# sys.path.append("pysrc")
# import pydevd


from train import numoflinks,numofparts,save_prefix

part_text = {
    0:("top head",13),#13
    1:("neck",14), #14
    2:("right shoulder",1),#1
    3:("right elbow",2), #2
    4:("right wrist",3), #3
    5:("left shoulder",4), #4
    6:("left elbow",5), #5
    7:("left wrist",6), #6
    8:("right hip",7), #7
    9:("right knee",8), #8
    10:("right ankle",9), #9
    11:("left hip",10),#10
    12:("left knee",11),#11
    13:("left ankle",12),#12
    14:("tummy"),#*
}


part_affinity = {
    
}



debug = False
# %matplotlib inline
import pylab as plt
#from generateLabelCPM import *
#from modelCPM import *
#sym = mxnetModule()

def showjson(json_path):
    import json,cv2
    try:
        ob = json.load(open(json_path,"rb"))
    except:
        print("warning",json_path)
        return
    img = cv2.imread(ob['path'])
    for key in ob.keys():
        if isinstance(ob[key],list) and len(ob[key]) == 30:
            for i in range(14):
                f_scale = 1.0/ max_img_shape[0] *max(img.shape[0],img.shape[1])
                x = int(ob[key][i * 2] * f_scale)
                y = int(ob[key][i * 2 + 1] * f_scale)
                cv2.circle(img,(x,y),3,(255,85,0))
                cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))               
                print(x,y)
    cv2.imshow("test",img)
    key = cv2.waitKey(0)
    if key == 27:
        exit(0)
search_ratio = [1]
imgshape_bind = [(int(368*x),int(368*x)) for x in search_ratio]
max_img_shape = (max(search_ratio)*368,max(search_ratio)*368)
import json
def padimg(img,destsize):
    s = img.shape
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
def parseOneJson(json_path):
    import json,cv2
    r_dict = {}

    try:
        ob = json.load(open(json_path,"rb"))
    except ValueError:
        print("warning", "invalid json file",json_path)
        return r_dict
    r_dict['image_id'] = os.path.splitext(os.path.basename(ob['path']))[0]

    img = cv2.imread(ob['path'])    
    keypoint_annotations = {}
    for key in ob.keys():
        if isinstance(ob[key],list) and len(ob[key]) == (2 * (numofparts-1)):
            keypoint = [0] * 42
            for i in range(numofparts-1):
                f_scale = 1.0/ max_img_shape[0] *max(img.shape[0],img.shape[1])
                x = int(ob[key][i * 2] * f_scale)
                y = int(ob[key][i * 2 + 1] * f_scale)
                v = 1
                map_index = i 
                if x > img.shape[0] and y > img.shape[1] :
                    x = 0
                    y = 0
                    v = 0
                if x==0 and y== 0:
                    x = 0
                    y = 0
                    v = 0
                keypoint[map_index *3 + 0] = x
                keypoint[map_index *3 + 1] = y
                keypoint[map_index *3 + 2] = v                                            
            keypoint_annotations['human{0}'.format(i)] = keypoint
    r_dict['keypoint_annotations'] = keypoint_annotations

    return r_dict
def parse_heatpaf(img_path,oriImg,heatmap_avg,paf_avg,output_json_prefix,im_show_cb=cv2.imshow):
    

    '''
    0：头顶
    1：脖子
    2：右肩
    3：右肘
    4：右腕

    '''
        

        
    param={}
 
    param['thre1'] = 0.2
    param['thre2'] = 0.1
    param['mid_num'] = 7


    import scipy
    print heatmap_avg.shape

    #plt.imshow(heatmap_avg[:,:,2])
    from scipy.ndimage.filters import gaussian_filter
    all_peaks = []
    peak_counter = 0

    
    for part in range(15-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
#         map = gaussian_filter(map_ori, sigma=3)
        map = map_ori
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[13, 14], [14, 1], [14, 4], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7], [7, 8],
            [8, 9], [4, 10], [10, 11], [11, 12]]
    # the middle joints heatmap correpondence
    mapIdx = [(i*2,i*2+1) for i in range(numoflinks)]
    assert(len(limbSeq) == numoflinks ) 

    connection_all = []
    special_k = []
    special_non_zero_index = []
    mid_num = param['mid_num'] 
    if debug:
        pydevd.settrace("115.154.62.162", True, True, 5678, True) 
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        # print(k)
        # print(candA)
        # print('---------')
        # print(candB)
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    # print('vec: ',vec)
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    # print('norm: ', norm)
                    vec = np.divide(vec, norm)
                    # print('normalized vec: ', vec)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    # print('startend: ', startend)
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                    for I in range(len(startend))])
                    # print('vec_x: ', vec_x)
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                    for I in range(len(startend))])
                    # print('vec_y: ', vec_y)
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    # print(score_midpts)
                    # print('score_midpts: ', score_midpts)
                    try:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    except ZeroDivisionError:
                        score_with_dist_prior = -1               
                    ##print('score_with_dist_prior: ', score_with_dist_prior)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    # print('score_midpts > param["thre2"]: ', len(np.nonzero(score_midpts > param['thre2'])[0]))
                    criterion2 = score_with_dist_prior > 0
                    
                    if criterion1 and criterion2:
                        # print('match')
                        # print(i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2])
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
                    # print('--------end-----------')
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # print('-------------connection_candidate---------------')
            # print(connection_candidate)
            # print('------------------------------------------------')
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    # print('----------connection-----------')
                    # print(connection)
                    # print('-------------------------------')
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        elif(nA != 0 or nB != 0):
            special_k.append(k)
            special_non_zero_index.append(indexA if nA != 0 else indexB)
            connection_all.append([])
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))

    candidate = np.array([item for sublist in all_peaks for item in sublist])
    print len(candidate)
    candidate


    for k in range(len(mapIdx)):
        if k not in special_k:
            try:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1
            except IndexError as e :
                row = -1 * np.ones(20)
                subset = np.vstack([subset, row])        
                continue
            except TypeError as e:
                row = -1 * np.ones(20)
                subset = np.vstack([subset, row])        
                continue
            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    ## Show human part keypoint

    # visualize
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # cmap = matplotlib.cm.get_cmap('hsv')

    # canvas = cv.imread(test_image) # B,G,R order
    # print len(all_peaks)
    # for i in range(15):
    #     rgba = np.array(cmap(1 - i/18. - 1./36))
    #     rgba[0:3] *= 255
    #     for j in range(len(all_peaks[i])):
    #         cv.circle(canvas, all_peaks[i][1][0:2], 4, colors[i], thickness=-1)

    # to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    # plt.imshow(to_plot[:,:,[2,1,0]])
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(11, 11)
    # # visualize 2
    import cv2
    stickwidth = 4
    if debug:
        canvas = oriImg
        img_ori = canvas.copy()
    output_path = output_json_prefix +  os.path.splitext(os.path.basename(img_path))[0] + ".json"
    with open(output_path,"wb") as f_json:
        r_all_person = {}
        r_all_person['path'] = img_path
        for n in range(len(subset)):
            r_oneperson = []
            for i in range(numofparts - 1):    
                index_head = subset[n][i]        
                # if -1 in index_head:
                #     continue
                x = int(candidate[index_head.astype(int),0])
                y = int(candidate[index_head.astype(int),1])
                coo = (x,y)
                if debug:
                    cv2.circle(img_ori,coo,3,colors[n],thickness = 3,)
                r_oneperson += coo
            r_all_person['human{0}'.format(n)]=r_oneperson           
        json.dump(r_all_person,f_json)
        if debug:
            print("[info]: finish writingcd ",output_path)
    if debug and im_show_cb:
        im_show_cb("result",img_ori)


def main(isdebug = False):
    debug = isdebug
    use_mpi_model = False
    import sys,cv2,os,argparse,copy
    from img2keypoint_using_ai import parse_heatpaf
    from pprint import pprint
    sys.path.append("/data1/yks/mxnet_ai/mxnet_pose_for_AI_challenger")
    from modelCPMWeight import CPMModel_test
    if not use_mpi_model:
        from train import numoflinks,numofparts,save_prefix
        start_epoch = 5250
    else:
        save_prefix = "../../models/openpose/realtimePose_mpi"
        start_epoch=0
        numofparts = 16
        numoflinks = 28
    import matplotlib.pyplot as plt
    #%matplotlib inline
    def imshow(x,y):
        fig = plt.gcf();fig.set_size_inches(8, 8);plt.title(x); plt.imshow(y);plt.show()
    import mxnet as mx
    import numpy as np
    max_img_shape = (368 ,368)
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
        sym_load, newargs, _ = mx.model.load_checkpoint(prefix, epoch)
        if use_mpi_model:
            sym = sym_load
        model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in gpus],                        
                              label_names=['heatmaplabel',
                                    'partaffinityglabel',
                                    'heatweight',
                                    'vecweight'])
        model.bind(data_shapes=[('data', (batch_size, 3, max_img_shape[0], max_img_shape[1]))],for_training = False)
        model.init_params(arg_params=newargs, aux_params={}, allow_missing=False)
        return model
    cmodel = getModel(save_prefix,start_epoch)
    # for x,y,z in os.walk("/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911"):
    for x,y,z in os.walk("../../eval_dataset/images/"):
        for name in z:
            img_path = os.path.join(x,name)
            pprint(img_path)
            img_path,oriImg,heatmap_avg,paf_avg = getHeatAndPAF(img_path,[cmodel])
#             imshow("heatmap_avg",np.float32( heatmap_avg[:,:,14]))
#             imshow("oriImg",oriImg)

            parse_heatpaf(img_path,oriImg,heatmap_avg,paf_avg,"../outputs/val3/",im_show_cb = imshow)
      
#             for i in range(heatmap_avg.shape[2]):
#                 imshow("heatmap_{0}".format(i),heatmap_avg[:,:,i] )
#             for i in range(paf_avg.shape[2]):
#                 paf = np.sqrt(paf_avg[:,:,i * 2].astype(np.float32) ** 2 +
#                               paf_avg[:,:,i *2 +1].astype(np.float32) **2) 
#                 print(paf.shape)
                
#                 imshow("paf_avg_{0}".format(i),paf)


        

if __name__ == "__main__":
    main()