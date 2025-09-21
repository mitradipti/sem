# Directory Structure
## Below is the directory structure
 ```
  dataset
    |-original_data
       |- 0_1.jpg
       |- 0_2.jpg
       |-...
    |-semantic_annotations_mask
       |- 0_1.png
       |- 0_2.png
       |-...
```

# Required Libraries
## 1. Pytorch 
    pip install torch torchvision
## 2. Transformers
     pip install transformers
## 3. Pillow
    pip install pillow
## 4. NumPy
    pip install numpy


# Train
The semantic_train.py file needs to be trained, and it is uploaded in the link. Need to adjust the dataset paths when running the code. After finishing training, a .pth file will be saved. 
