#!/usr/bin/python2
#encoding=utf-8

import cv2,json
import sys
import argparse
sys.path.append("ai_openpose")
from img2keypoint_using_ai import padimg,parseOneJson
from pprint import pprint
from random import randint
import shutil
import os,sys

default_json_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"
default_images_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911"

def random_sample(maxcount = 100,json_path = default_json_path,images_path = default_images_path):
    import json,os,sys
    obj = json.load(open(default_json_path,"rb"))
    obj_output = []
    for _ in range(min(maxcount,len(obj))):
        index = randint(0,len(obj))
        image_id = obj[index]['image_id']
        srcpath = default_images_path + "/{0}.jpg".format(image_id)
        destpath = "../..//eval_dataset/images/"
        shutil.copy2(srcpath,destpath)
        obj_output.append(obj[index]) 
        del obj[index]
    json.dump(obj_output,open("../../eval_dataset/eval.json","wv"))
    
def mergejson(input_dir):
    r = []
    for x, y, z in os.walk(input_dir):
        for name in z:
            if str(name).endswith(".json"):
                r.append(parseOneJson(os.path.join(x,name)))
    return r

def eval_model():
    import img2keypoint_using_ai
    img2keypoint_using_ai.main(start_epoch = 8000)
    with open("../outputs/val3.json","wb") as f:
        json.dump(mergejson("../outputs/val3/"),f)
    import keypoint_eval
    keypoint_eval.main("../../eval_dataset/eval.json","../outputs/val3.json")
if __name__ == "__main__":
    eval_model()
    

