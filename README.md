# Wheelchair Detection

## 1. Custom Dataset

### 1.1 Assortment Directories

```bash
[datasets] ---+--- [date-20230821] ------+ d00000.jpg
              |                          + d00001.jpg
              |                          + ...
              |
              +--- [kaggle]        ------+ k00000.jpg
              |                          + k00001.jpg
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
