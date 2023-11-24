#!/bin/bash -e

##############################
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

##############################
NAME=yolo-fastest
#NAME=yolov3-tiny
CFG="cfg/${NAME}.cfg"
GPUS="-gpus 0"
WEIGHTS=""

WIDTH=$(cat ${CFG} | grep "^width" | awk -F '=' '{print $2}')
HEIGHT=$(cat ${CFG} | grep "^height" | awk -F '=' '{print $2}')

function append_train_test_list () {
	local D=$1
	local E=$2
	local N=0
	local R=0
	for F in `find $(pwd)/datasets/${D} -name '*.txt'` ; do
		R=$(($N % 10))
		if [ ${R} -eq 1 ]; then echo ${F} | sed "s|.txt$|.${E}|" >> test.txt ; else echo ${F} | sed "s|.txt$|.${E}|" >> train.txt ; fi
		N=$(($N + 1))
	done
}

##############################
rm -rf train.txt test.txt predictions.jpg data

append_train_test_list mobilityaids png
append_train_test_list roboflow jpg
append_train_test_list wheelchair jpg
append_train_test_list person jpg
append_train_test_list date-20230821 jpg
append_train_test_list kaggle jpg

##############################
sed "s|/work/Yolo-Fastest/wheelchair|`pwd`|" -i cfg/${NAME}.data

##############################
echo '' && echo -e "${YELLOW} echo '' | ../darknet detector calc_anchors cfg/${NAME}.data -num_of_clusters 6 -width ${WIDTH} -height ${HEIGHT} -dont_show ${NC}"
echo '' | ../darknet detector calc_anchors cfg/${NAME}.data -num_of_clusters 6 -width ${WIDTH} -height ${HEIGHT} -dont_show
[ 0 -ne $(cat ${CFG} | grep "^anchors" | awk -F '=' '{print $2}' | wc -l) ] && cat ${CFG} | grep "^anchors" | awk -F '=' '{print $2}' | tail -1 > cfg/${NAME}.anchors

##############################
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib
export PATH=/usr/local/cuda/bin:${PATH}

##############################
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup

[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
echo ""
echo -e "${YELLOW} ../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map ${NC}"
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py ]; then
	git -C ../keras-YOLOv3-model-set checkout tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	sed "s|model_input_shape = \"160x160\"|model_input_shape = \"${WIDTH}x${HEIGHT}\"|" -i ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py

	echo ""
	echo -e "${YELLOW} python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py --config_path cfg/${NAME}.cfg --weights_path backup/${NAME}_final.weights --output_path backup/${NAME}.h5 ${NC}"
	rm -rf backup/${NAME}.h5
	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
		--config_path cfg/${NAME}.cfg \
		--weights_path backup/${NAME}_final.weights \
		--output_path backup/${NAME}.h5 || true

	echo ""
	echo -e "${YELLOW} python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py --keras_model_file backup/${NAME}.h5 --annotation_file train.txt --output_file backup/${NAME}.tflite ${NC}"
	[ -e backup/${NAME}.h5 ] && \
		python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py \
		--keras_model_file backup/${NAME}.h5 \
		--annotation_file train.txt \
		--output_file backup/${NAME}.tflite

	echo ""
	echo -e "${YELLOW} xxd -i backup/${NAME}.tflite > backup/${NAME}.cc ${NC}"
	[ -e backup/${NAME}.tflite ] && \
		xxd -i backup/${NAME}.tflite > backup/${NAME}.cc
fi

##############################
echo ""
echo -e "${YELLOW} ../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 ${NC}"
../darknet detector map cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights -iou_thresh 0.5 | grep -v '\-points'

##############################
echo ""
echo -e "${YELLOW} Detector Test: ${NC}"
echo -e "${YELLOW} ../darknet detector test cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show ${NC}"
echo ""
exit 0
