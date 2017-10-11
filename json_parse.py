import cv2,json
import sys
import argparse
from img2keypoint_using_coco import padimg,parseOneJson

def mergejson(input_dir):
    r = []
    for x, y, z in os.walk(args.input):
        for name in z:
            if str(name).endswith(".json"):
                r.append(parseOneJson(os.path.join(x,name)))
    return r

if __name__ == "__main__":
    import os
    testpath = "/data/yks/ai_challenger/ai_challenger_keypoint_test_a_20170923/keypoint_test_a_images_20170923_json/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i", help="input json",type = str,default = testpath)
    parser.add_argument("--output","-o", help="output json",type = str,default = "output.json")    
    args = parser.parse_args()
    with open(args.output,"wb") as f:
        json.dump(mergejson(args.input),f)
