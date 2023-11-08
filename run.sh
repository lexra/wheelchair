#!/bin/bash -e

##############################
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

##############################
NAME=yolo-wheelchair
#NAME=yolov3-wheelchair
#NAME=yolov3-tiny
CFG="cfg/${NAME}.cfg"
GPUS="-gpus 0"
WEIGHTS=""

INPUT_W=$(cat ${CFG} | grep width | awk -F '=' '{print $2}')
INPUT_H=$(cat ${CFG} | grep height | awk -F '=' '{print $2}')

function append_train_test_list () {
	local D=$1
	local E=$2
	local N=0
	local R=0
	for F in `find $(pwd)/datasets/${D} -name '*.txt'` ; do
		R=$(($N % 10))
		if [ ${R} -eq 1 ]; then echo ${F} | sed "s|.txt$|.${E}|" ; echo ${F} | sed "s|.txt$|.${E}|" >> test.txt ; else echo ${F} | sed "s|.txt$|.${E}|" ; echo ${F} | sed "s|.txt$|.${E}|" >> train.txt ; fi
		N=$(($N + 1))
	done
}

##############################
rm -rfv train.txt test.txt

append_train_test_list mobilityaids png
append_train_test_list roboflow jpg
append_train_test_list wheelchair jpg
append_train_test_list person jpg
append_train_test_list date-20230821 jpg
append_train_test_list kaggle jpg

##############################
sed "s|/work/Yolo-Fastest/wheelchair|`pwd`|" -i cfg/${NAME}.data
[ 0 -ne $(cat ${CFG} |grep anchors | awk -F '=' '{print $2}' | wc -l) ] && cat ${CFG} |grep anchors | awk -F '=' '{print $2}' | tail -1 > cfg/${NAME}.anchors

##############################
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup

[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py ]; then
	git -C ../keras-YOLOv3-model-set checkout tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	sed "s|model_input_shape = \"160x160\"|model_input_shape = \"${INPUT_W}x${INPUT_H}\"|" -i ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py

	echo "python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py --config_path cfg/${NAME}.cfg --weights_path backup/${NAME}_final.weights --output_path backup/${NAME}.h5"
	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py --config_path cfg/${NAME}.cfg --weights_path backup/${NAME}_final.weights --output_path backup/${NAME}.h5 || true

	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py \
		--keras_model_file backup/${NAME}.h5 \
		--annotation_file train.txt --output_file \
		backup/${NAME}.tflite || true
	#xxd -i backup/${NAME}.tflite > backup/${NAME}-$(date +'%Y%m%d').cc || true
	xxd -i backup/${NAME}.tflite > backup/${NAME}.cc || true
fi

##############################
../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 | grep -v '\-points'

##############################
echo ""
echo -e "${YELLOW} Detector Test: ${NC}"
echo -e "${YELLOW} ../darknet detector test cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show ${NC}"
echo ""
exit 0
