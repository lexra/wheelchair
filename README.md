# Wheelchair Detection

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

Here we use the `YoloLabel` (https://github.com/developer0hye/Yolo_Label) for Bounding Box Labeling of a given Assortment Directory. 

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

As the picture above, The Bounding Box txt file, `kaggle/00314.txt`, is generated accordingly complied with the `kaggle/00314.jpg`. 

## 2. Generating Train List and Test List 

### 2.1 function append_train_test_list ()

```bash
function append_train_test_list () {
        local D=$1; local E=$2; local N=0; local R=0;

        for F in `find $(pwd)/datasets/${D} -name '*.txt'` ; do
                R=$(($N % 10))
                if [ ${R} -eq 1 ]; then
                        echo ${F} | sed "s|.txt$|.${E}|" >> test.txt
                else echo ${F} | sed "s|.txt$|.${E}|"
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

Test List occupied one-tenth, and Train List occupied nine-tenth. 

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

#### 3.3.3 classes

```python
...
classes=2
...
classes=2
...
```










