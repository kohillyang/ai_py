eval:

	python2 json_parse.py -i ../outputs/val3 -o ../outputs/val3.json
	python keypoint_eval.py --ref ../../eval_dataset/eval.json --submit ../outputs/val3.json 
img2keypoint:
	-rm ../outputs/val3/*
	python2 img2keypoint_using_ai.py

