#!/usr/bin/python2
#encoding=utf-8
def padimg(img,destsize):
    import cv2
    import numpy as np
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
def convert2json():
    import scipy.io as sio
    dmpi = sio.loadmat("mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat", struct_as_record=False)
    import json
    all_img = []
    for img_index in range(dmpi['RELEASE'][0,0].annolist.shape[1]):
        img_name = dmpi['RELEASE'][0,0].annolist[0,img_index].image[0,0].name[0]
        one_img = {}
        one_img['img_path'] = "/data1/yks/dataset/openpose_dataset/mpi/images/" + img_name
        one_img['annoations'] = []
        for rect_index in range(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect.shape[1]):
            one_rect = {}
            one_rect['annopoints'] = []

            try:
                one_rect['x1'] = int(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].x1[0,0])
                one_rect['y1'] = int(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].y1[0,0])
                one_rect['x2'] = int(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].x2[0,0])
                one_rect['y2'] = int(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].y2[0,0])
                one_rect['scale'] = float(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].scale[0,0])        
              
                for point_index in range(dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].annopoints[0,0].point.shape[1]):
                    try:
                        x = dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].annopoints[0,0].point[0,point_index].x[0,0]
                        y = dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].annopoints[0,0].point[0,point_index].y[0,0]
                        part_id = dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].annopoints[0,0].point[0,point_index].id[0,0]
                        is_visible = dmpi['RELEASE'][0,0].annolist[0,img_index].annorect[0,rect_index].annopoints[0,0].point[0,point_index].is_visible[0,0]
                        print(point_index,x,y,part_id,is_visible)
                        one_rect['annopoints'].append([int(x),int(y),int(part_id),bool(is_visible)])                
                    except Exception as e:
                        print("                              ",e)
            except AttributeError as e:
                print e
            except IndexError as e:
                print e
            if len(one_rect['annopoints']) == 14:
                one_img['annoations'].append(one_rect)
        if len(one_img['annoations']) > 0:
            all_img.append(one_img)
    print(len(all_img))
    with open("a.json","wb") as f:
        json.dump(all_img,f)

def show_mpi_json():
    import json,cv2
    all_img = json.load(open("a.json","rb"))[2664:]
    for oneimg in all_img:
        img = cv2.imread(oneimg['img_path'])

        for one_rect in oneimg['annoations']:
            cv2.rectangle(img,(one_rect['x1'],one_rect['y1']),(one_rect['x2'],one_rect['y2']),(0,255,255),-1,8)
            for point in one_rect['annopoints']:
                cv2.circle(img,(point[0],point[1]),12,(0,255,0),-1,8)
                print(point[2])
        img = padimg(img,480)
        cv2.imshow("img",img)
        if cv2.waitKey(0) == 27:
            exit(0)

if __name__ == "__main__":

    #convert2json()

    show_mpi_json()
#dmpi['RELEASE'][0,0].annolist.shape
#dmpi['RELEASE'][0,0].annolist[0,82].image[0,0].name[0]
#dmpi['RELEASE'][0,0].annolist[0,82].annorect.shape
#dmpi['RELEASE'][0,0].annolist[0,2665].annorect[0,0].annopoints[0,0].point.shape
#dmpi['RELEASE'][0,0].annolist[0,2665].annorect[0,0].annopoints[0,0].point[0,3].x[0,0]