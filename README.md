# ObjectClicker
Simple user interface to count objects with Segment Anything. Meant as a first pass for anyone with messy data. Allows easy bootstrapping of object classification data to be used in future models

# Install 
No extra install instructions past install SAM2 using facebooks instructions here https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md

Make sure to download the correct model and model.yaml file which can be found https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb or more simply

wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://raw.githubusercontent.com/facebookresearch/segment-anything-2/refs/heads/main/sam2_configs/sam2_hiera_l.yaml



# Example use Case 1: Cell Annotation
Getting cell counts on irregualr images
![image](https://github.com/user-attachments/assets/3fe2a50b-c49b-402f-989a-9cfde0e9df73)

