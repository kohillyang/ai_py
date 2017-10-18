#!/usr/bin/python2
import cv2,json
import sys
import argparse
from img2keypoint_using_ai import padimg,showjson

if __name__ == "__main__":
    import os
    testpath = "result/val0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i", help="input json",type = str,default = testpath)
    args = parser.parse_args()
    # showjson('/data/yks/ai_challenger/ai_challenger_keypoint_test_a_20170923/keypoint_test_a_images_20170923_json/1a56c170a488c81870e6e438ac550eedc2a9e441.json')
    for x, y, z in os.walk(args.input):
        for name in z:
            if str(name).endswith(".json"):
                json_path = os.path.join(x,name)
                print(json_path)
                showjson(json_path)
  
