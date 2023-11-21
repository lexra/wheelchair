# Dataset for YOLOv3 Wheelchair Detection

<img src=https://github.com/lexra/wheelchair/assets/33512027/e61e8897-96de-4653-a72b-3c2bcf98a21b width=600 />


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

Here the format of the annotation is: 

```
<class> <x> <y> <width> <height> 
ex: 0 0.25 0.44 0.5 0.8
class is the object class, (x,y) are centre coordinates of the bounding box.
width, height represent width and height of the bounding box. 
```

And we use the <a href=https://github.com/developer0hye/Yolo_Label>YoloLabel</a> for Bounding Box Labeling of a given Assortment Directory. 


<img src=https://github.com/lexra/wheelchair/assets/33512027/bd262a8b-75ac-4e5a-9b45-497bb62422d0 width=600/>

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

### 3.1 Yolov3-tiny.name

```bash
person
wheelchair
```

### 3.2 Yolov3-tiny.data

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

### 3.3 Yolov3-tiny.cfg

Download the `yolov3-tiny.cfg` from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg . 

```python
# Testing(此处下面的两行，测试的时候开启即可)
#batch=1                 # 每batch个样本更新一次参数。
#subdivisions=1          # 如果内存不够大，将batch分割为subdivisions个子batch，每个子batch的大小为batch/subdivisions。

# Training(此处下面的两行，训练的时候开启即可)
batch=64                 # 表示网络积累多少个样本后进行一次正向传播
subdivisions=16          # 将一个batch的图片分sub次完成网络的正向传播

width=416                # 输入图像的宽
height=416               # 输入图像的高
channels=3               # 输入图像的通道数
momentum=0.9             # 动量系数
decay=0.0005             # 权重衰减正则项，防止过拟合

# 下面四行，是数据增强的参数
angle=0                  # 通过旋转角度来生成更多训练样本
saturation = 1.5         # 通过调整饱和度来生成更多训练样本
exposure = 1.5           # 通过调整曝光量来生成更多训练样本
hue=.1                   # 通过调整色调来生成更多训练样本

learning_rate=0.001      # 初始学习率
burn_in=1000             #
max_batches = 500200     # 训练达到max_batches后停止学习
policy=steps             # 调整学习率的policy，有如下policy：CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
steps=400000,450000      # 根据batch_num调整学习率
scales=.1,.1             # 学习率变化的比例，累计相乘

[convolutional]
batch_normalize=1        # 是否做BN
filters=32               # 卷积核的个数，也是输出的特征图的维度
size=3                   # 卷积核的尺寸3*3
stride=1                 # 做卷积运算的步长
pad=1                    # 如果pad为0,padding由 padding参数指定。如果pad为1，padding大小为size/2
activation=leaky
```

#### 3.3.1 Channels

```python
channels=3
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

#### 3.3.4 Classes

```python
classes=2
```

Note: the classes value here must precisely conform to the one in `Yolov3-tiny.data`. 

#### 3.3.5 Filters

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

#### 3.3.6 Mask

```python
[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
```

Here only the 3/4/5 pairs , `81,82,  135,169,  344,319`, work. 

```python
[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
```

And here only the 0/1/2 pairs , `10,14,  23,27,  37,58`, work. 

#### 3.3.6 max_batches

```python
max_batches = (classes * 2000)
```

In order to save time, we could modify the `max_batches` according to the formula described above. 


## 4. Anchors

### 4.1 Calc_anchors

```bash
 echo '' | ../darknet detector calc_anchors cfg/yolov3-tiny.data -num_of_clusters 6 -width 416 -height 416 -dont_show
```

```bash
 CUDA-version: 12020 (12020), cuDNN: 8.9.6, CUDNN_HALF=1, GPU count: 1
 CUDNN_HALF=1
 OpenCV version: 4.2.0
 num_of_clusters = 6, width = 416, height = 416
 read labels from 1590 images
 loaded          image: 1590     box: 2779
 all loaded.
 calculating k-means++ ...
 iterations = 32

counters_per_class = 1228, 1551
 avg IoU = 76.29 %

Saving anchors to the file: anchors.txt
anchors =  50,158,  70,274, 123,222, 116,344, 193,312, 300,359
```

* -num_of_clusters: 需要几组. tiny的6组, yolo的9组.
* YOLOv3 的 `anchor` 是是相对于输入图片的, 比如 320x224 的图片. 

### 4.2 <a href=https://zhuanlan.zhihu.com/p/338147028>先驗框</a> 的作用

<img src=https://pic2.zhimg.com/v2-7472c86f84a363e23758f73cb2bfff8d_r.jpg />

左图我们更希望模型选择红色的先验框，右图希望模型选择蓝色的先验框，这样使得模型更容易学习. 

## 5. Train, Test

### 5.1 Train

```bash
../darknet detector train cfg/yolov3-tiny.data cfg/yolov3-tiny.cfg \
    backup/yolov3-tiny_last.weights -gpus 0 -dont_show -map
```

```bash
...
Total BFLOPS 0.054
avg_outputs = 15205
 Allocate additional workspace_size = 52.73 MB
Loading weights from backup/yolo-wheelchair_last.weights...
 seen 64, trained: 16006 K-images (250 Kilo-batches_64)
Done! Loaded 131 layers from weights-file
Learning Rate: 0.001, Momentum: 0.949, Decay: 0.0005
 Detection layer: 121 - type = 28
 Detection layer: 130 - type = 28
Saving weights to /work/Yolo-Fastest/wheelchair/backup/yolo-wheelchair_final.weights
```

After training completed, `backup/yolov3-tiny_final.weights` is generated. 

### 5.1.1 Checkpoints

```bash
regfae@regulus-ASUS:/work/Yolo-Fastest/wheelchair$ ls -l backup/yolov3-tiny*
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 09:07 backup/yolov3-tiny_100000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 03:49 backup/yolov3-tiny_10000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 09:43 backup/yolov3-tiny_110000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 10:18 backup/yolov3-tiny_120000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 10:53 backup/yolov3-tiny_130000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 11:29 backup/yolov3-tiny_140000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 12:04 backup/yolov3-tiny_150000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 12:39 backup/yolov3-tiny_160000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 13:15 backup/yolov3-tiny_170000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 13:51 backup/yolov3-tiny_180000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 14:27 backup/yolov3-tiny_190000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 15:02 backup/yolov3-tiny_200000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 04:24 backup/yolov3-tiny_20000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 15:38 backup/yolov3-tiny_210000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 16:14 backup/yolov3-tiny_220000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 16:50 backup/yolov3-tiny_230000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 17:26 backup/yolov3-tiny_240000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 18:02 backup/yolov3-tiny_250000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 18:37 backup/yolov3-tiny_260000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 19:13 backup/yolov3-tiny_270000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 19:48 backup/yolov3-tiny_280000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 20:24 backup/yolov3-tiny_290000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 21:00 backup/yolov3-tiny_300000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 04:59 backup/yolov3-tiny_30000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 21:35 backup/yolov3-tiny_310000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 22:11 backup/yolov3-tiny_320000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 22:47 backup/yolov3-tiny_330000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 23:22 backup/yolov3-tiny_340000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 23:58 backup/yolov3-tiny_350000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 00:33 backup/yolov3-tiny_360000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 01:08 backup/yolov3-tiny_370000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 01:44 backup/yolov3-tiny_380000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 02:20 backup/yolov3-tiny_390000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 02:56 backup/yolov3-tiny_400000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 05:35 backup/yolov3-tiny_40000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 03:31 backup/yolov3-tiny_410000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 04:07 backup/yolov3-tiny_420000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 04:43 backup/yolov3-tiny_430000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 05:18 backup/yolov3-tiny_440000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 05:54 backup/yolov3-tiny_450000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 06:29 backup/yolov3-tiny_460000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 07:06 backup/yolov3-tiny_470000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 07:41 backup/yolov3-tiny_480000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 08:17 backup/yolov3-tiny_490000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 08:52 backup/yolov3-tiny_500000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 06:11 backup/yolov3-tiny_50000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 06:46 backup/yolov3-tiny_60000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 07:21 backup/yolov3-tiny_70000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 07:56 backup/yolov3-tiny_80000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 08:32 backup/yolov3-tiny_90000.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 18 07:34 backup/yolov3-tiny_best.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 21 16:33 backup/yolov3-tiny_final.weights
-rw-rw-r-- 1 regfae regfae 34714236 Nov 19 08:53 backup/yolov3-tiny_last.weights
```

### 5.2  mean Average Precision (mAP@0.50)

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

### 5.3  Test

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

## Appendix

### 1. CUDA Forward Compatible Upgrade / NVIDIA Kernel Mode Driver

![image](https://github.com/lexra/wheelchair/assets/33512027/081048bf-1190-4194-8821-9065b893bf8c)

### 2. Nvidia-driver-535 + cuda 12.2 + cudnn 8.9 for Ubuntu-20.0.4

#### 2.1 cuDNN v8.9.5 Download

Download cuDNN v8.9.5 (cudnn-local-repo-ubuntu2004-8.9.5.30_1.0-1_amd64.deb)

#### 2.2 Nvidia-driver Installation

```bash
sudo apt update && sudo apt install -y nvidia-driver-535
```

#### 2.3 Upgrade

```bash
sudo apt ugrade || sudo dpkg -i --force-overwrite \
    /var/cache/apt/archives/nvidia-kernel-common-535_535.129.03-0ubuntu1_amd64.deb
```

The upgrade would be failed; use `sudo dpkg -i --force-overwrite ...` to resolve this bug. 

#### 2.4 Cuda-toolkit Installation

```bash
sudo apt install -y cuda-compat-12-2 cuda-toolkit-12-2
```

Note: do not install the `nvidia-cuda-toolkit`; if already installed, use the following to remove: 

```
apt autoremove nvidia-cuda-toolkit -- purge
```

#### 2.5 cuDNN Installation

```
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.5.30_1.0-1_amd64.deb
```

```
sudo apt install -y libcudnn8
```

#### 2.5 Nccl Installation

```
sudo apt install -y libnccl2
```

### 3. Mobility-aids Pushing-Wheelchair

#### 3.1 People and Their Mobility Aids

```bash
http://mobility-aids.informatik.uni-freiburg.de/
```

#### 3.2 Person Guidance Scenario: Deep Detection of People and their Mobility Aids for a Hospital Robot

```bash
https://www.youtube.com/watch?v=X8HGhFUgquk
```

#### 3.3 Deep Detection of People and their Mobility Aids for a Hospital Robot

```bash
https://youtu.be/uQRnllNBcfU
```

### 4. Miniconda

#### 4.1 Download

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### 4.2 Miniconda Install

```
bash ./Miniconda3-latest-Linux-x86_64.sh
// 安裝完可以把他刪了
rm ./Miniconda3-latest-Linux-x86_64.sh
```

#### 4.3 PATH

```
vim ~/.bashrc
export PATH="$HOME/miniconda3/bin":$PATH
```

#### 4.4 Create a new `ENV_NAME`

```
conda create -n ENV_NAME python=3.7
```

#### 4.4 Activate / Deactivate `ENV_NAME`

```
//激活環境，此時可以安裝你需要的套件
conda activate ENV_NAME
//退出環境
conda deactivate
```

#### 4.5 Package Install

```
//例如安裝資料處理常用的pandas
conda install pandas
//或是
pip install pandas
```

#### 4.5 Remove `ENV_NAME`

```
conda remove --name ENV_NAME --all
```









