# Dataset for YOLOv3 Wheelchair Detection

## 1. Custom Dataset

### 1.1 Assortment Directories

```bash
[datasets] ---+--- [date-20230821] ------+ d00000.jpg
              |                          + d00001.jpg
              |                          + ...
              |
              +--- [kaggle]        ------+ 00000.jpg
              |                          + 00001.jpg
              |                          + ...
              |
              +--- [mobilityaids]  ------+ m00000.png
              |                          + m00001.png
              |                          + ...
              |
              +--- [person]        ------+ p00000.jpg
              |                          + p00001.jpg
              |                          + ...
              |
              +--- [roboflow]      ------+ p00000.jpg
              |                          + p00001.jpg
              |                          + ...
              |
              +--- [wheelchair]    ------+ w00000.jpg
                                         + w00001.jpg
                                         + ...
```

### 1.2 Bounding Box Labeling

Here we use the <a href=https://github.com/developer0hye/Yolo_Label>YoloLabel</a> for Bounding Box Labeling of a given Assortment Directory. 


<img src=https://github.com/lexra/wheelchair/assets/33512027/bd262a8b-75ac-4e5a-9b45-497bb62422d0 width=800/>

```bash
ls -l datasets/kaggle/00314.jpg
-rwxrwxr-x 1 regfae regfae 53505 Nov 18 02:17 datasets/kaggle/00314.jpg
```

```bash
cat datasets/kaggle/00314.txt
1 0.229654 0.543033 0.169454 0.274590
0 0.323300 0.486680 0.115942 0.403689
0 0.775362 0.501025 0.154961 0.375000
```

As the picture, `kaggle/00314.jpg`, above, The Bounding Box txt file, `kaggle/00314.txt`, is generated accordingly. 

The format of the annotation is: 

```
<class> <x> <y> <width> <height> 
ex: 0 0.25 0.44 0.5 0.8
class is the object class, (x,y) are centre coordinates of the bounding box. width, height represent width and height of the bounding box
```

## 2. Generating Train List and Test List 

### 2.1 function append_train_test_list ()

```bash
function append_train_test_list () {
        local D=$1; local E=$2; local N=0; local R=0;

        for F in `find $(pwd)/datasets/${D} -name '*.txt'` ; do
                R=$(($N % 10))
                if [ ${R} -eq 1 ]; then
                        echo ${F} | sed "s|.txt$|.${E}|" >> test.txt
                else
                        echo ${F} | sed "s|.txt$|.${E}|" >> train.txt
                fi
                N=$(($N + 1))
        done
}
```

The first argument to append_train_test_list() is `the Assortment Directory`; the second is the pictures file extension in that given Directory. 

### 2.2 Regenerating Train List and Test List 

```bash
rm -rfv train.txt test.txt

append_train_test_list mobilityaids png
append_train_test_list roboflow jpg
append_train_test_list wheelchair jpg
append_train_test_list person jpg
append_train_test_list date-20230821 jpg
append_train_test_list kaggle jpg
```

Test List occupied `10%`, and Train List occupied `90%`. 

## 3. yolov3-tiny.cfg, yolov3-tiny.data, yolov3-tiny.name

### 3.1 yolov3-tiny.name

```bash
person
wheelchair
```

### 3.2 yolov3-tiny.data

```bash
mkdir -p backup
```

```bash
ln -s ../data .
```

```bash
classes=2
train=/work/Yolo-Fastest/wheelchair/train.txt
valid=/work/Yolo-Fastest/wheelchair/test.txt
names=/work/Yolo-Fastest/wheelchair/cfg/yolov3-tiny.names
backup=/work/Yolo-Fastest/wheelchair/backup
```

### 3.3 yolov3-tiny.cfg

#### 3.3.1 Download

```bash
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
```

#### 3.3.2 Batch, Subdivisions

```python
batch=32
subdivisions=1
```

#### 3.3.3 Width, Height

```python
width=416
height=416
```

#### 3.3.4 classes

```python
classes=2
```

#### 3.3.5 filters

Calculate filters according to the following formula: `filters=21`

```python
filters = (classes + 5) * 3
```

There are 2 pairs of `[yolo]` brackets for `yolov3-tiny.cfg`; the only 2, `filters=`, that we need modify locate just above and nearest the `[yolo]` brackets. 

```python
filters=21

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=2
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

```python
filters=21

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=2
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

#### 3.3.6 mask

```python
[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
```

which means only the 3/4/5 pairs , `81,82,  135,169,  344,319`, work. 

```python
[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
```

only the 0/1/2 pairs , `10,14,  23,27,  37,58`, work.


## 4. Train, Test

### 4.1 Train

```bash
../darknet detector train cfg/yolov3-tiny.data cfg/yolov3-tiny.cfg \
    backup/yolov3-tiny_last.weights -gpus 0 -dont_show -map
```

### 4.2  mean Average Precision (mAP@0.50)

```bash
../darknet detector map cfg/yolov3-tiny.data cfg/yolov3-tiny.cfg \
    backup/yolov3-tiny_final.weights -iou_thresh 0.5
```

```bash
 CUDA-version: 12020 (12020), cuDNN: 8.9.6, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 890, cudnn_half = 1, GPU: NVIDIA GeForce RTX 4070 Laptop GPU
net.optimized_memory = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 Create CUDA-stream - 0
 Create cudnn-handle 0
conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
   1 max                2x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16 0.003 BF
   2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32 0.399 BF
   3 max                2x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32 0.001 BF
   4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64 0.399 BF
   5 max                2x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64 0.001 BF
   6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128 0.399 BF
   7 max                2x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128 0.000 BF
   8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256 0.399 BF
   9 max                2x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256 0.000 BF
  10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  11 max                2x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.000 BF
  12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
  13 conv    256       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 256 0.089 BF
  14 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  15 conv     21       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  21 0.004 BF
  16 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
  17 route  13                                     ->   13 x  13 x 256
  18 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  19 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  20 route  19 8                                   ->   26 x  26 x 384
  21 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  22 conv     21       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  21 0.007 BF
  23 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
Total BFLOPS 5.449
avg_outputs = 325057
 Allocate additional workspace_size = 19.91 MB
Loading weights from backup/yolov3-tiny_final.weights...
 seen 64, trained: 16006 K-images (250 Kilo-batches_64)
Done! Loaded 24 layers from weights-file

 calculation mAP (mean average precision)...
 Detection layer: 16 - type = 28
 Detection layer: 23 - type = 28
180
 detections_count = 330, unique_truth_count = 309
class_id = 0, name = person, ap = 90.11%         (TP = 107, FP = 2)
class_id = 1, name = wheelchair, ap = 96.62%     (TP = 172, FP = 2)

 for conf_thresh = 0.25, precision = 0.99, recall = 0.90, F1-score = 0.94
 for conf_thresh = 0.25, TP = 279, FP = 4, FN = 30, average IoU = 86.63 %

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
 mean average precision (mAP@0.50) = 0.933629, or 93.36 %
```

### 4.3  Test

```bash
../darknet detector test cfg/yolov3-tiny.data cfg/yolov3-tiny.cfg \
    backup/yolov3-tiny_final.weights pixmaps/push_wheelchair.jpg -ext_output -dont_show
```

```
 CUDA-version: 12020 (12020), cuDNN: 8.9.6, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 0 : compute_capability = 890, cudnn_half = 1, GPU: NVIDIA GeForce RTX 4070 Laptop GPU
net.optimized_memory = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 Create CUDA-stream - 0
 Create cudnn-handle 0
conv     16       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  16 0.150 BF
   1 max                2x 2/ 2    416 x 416 x  16 ->  208 x 208 x  16 0.003 BF
   2 conv     32       3 x 3/ 1    208 x 208 x  16 ->  208 x 208 x  32 0.399 BF
   3 max                2x 2/ 2    208 x 208 x  32 ->  104 x 104 x  32 0.001 BF
   4 conv     64       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  64 0.399 BF
   5 max                2x 2/ 2    104 x 104 x  64 ->   52 x  52 x  64 0.001 BF
   6 conv    128       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x 128 0.399 BF
   7 max                2x 2/ 2     52 x  52 x 128 ->   26 x  26 x 128 0.000 BF
   8 conv    256       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 256 0.399 BF
   9 max                2x 2/ 2     26 x  26 x 256 ->   13 x  13 x 256 0.000 BF
  10 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  11 max                2x 2/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.000 BF
  12 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
  13 conv    256       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 256 0.089 BF
  14 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  15 conv     21       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  21 0.004 BF
  16 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
  17 route  13                                     ->   13 x  13 x 256
  18 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  19 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  20 route  19 8                                   ->   26 x  26 x 384
  21 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  22 conv     21       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  21 0.007 BF
  23 yolo
[yolo] params: iou loss: mse (2), iou_norm: 0.75, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
Total BFLOPS 5.449
avg_outputs = 325057
 Allocate additional workspace_size = 19.91 MB
Loading weights from backup/yolov3-tiny_final.weights...
 seen 64, trained: 16006 K-images (250 Kilo-batches_64)
Done! Loaded 24 layers from weights-file
 Detection layer: 16 - type = 28
 Detection layer: 23 - type = 28
pixmaps/push_wheelchair.jpg: Predicted in 25.849000 milli-seconds.
person: 100%    (left_x:  191   top_y:  103   width:  267   height:  726)
wheelchair: 100%        (left_x:  334   top_y:  233   width:  627   height:  639)
```








