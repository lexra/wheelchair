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
              +--- [mobilityaids]  ------+ m00000.jpg
              |                          + m00001.jpg
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

### 1.2 Bounding Box Labeling Tool

Here we use the `YoloLabel` (https://github.com/developer0hye/Yolo_Label) for Bounding Box Labeling of a given Assortment Directory. 

<img src=https://github.com/lexra/wheelchair/assets/33512027/bd262a8b-75ac-4e5a-9b45-497bb62422d0 width=800/>

```bash
ls -l datasets/kaggle
total 20636
-rw-rw-r-- 1 regfae regfae  14805 Nov 18 02:17 00000.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00000.txt
-rw-rw-r-- 1 regfae regfae  22666 Nov 18 02:17 00001.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00001.txt
-rw-rw-r-- 1 regfae regfae  27163 Nov 18 02:17 00002.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00002.txt
-rw-rw-r-- 1 regfae regfae  26984 Nov 18 02:17 00003.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00003.txt
-rw-rw-r-- 1 regfae regfae  37373 Nov 18 02:17 00004.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00004.txt
-rw-rw-r-- 1 regfae regfae  20670 Nov 18 02:17 00005.jpg
-rw-rw-r-- 1 regfae regfae     39 Nov 18 02:17 00005.txt
-rw-rw-r-- 1 regfae regfae  25998 Nov 18 02:17 00006.jpg
...
```
