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

