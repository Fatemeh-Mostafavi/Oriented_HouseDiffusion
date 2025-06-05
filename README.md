# Oriented_HouseDiffusion
Orientation-aware Floor Layout Generation using Diffusion Models

![image](https://github.com/user-attachments/assets/85b72d42-06b7-42a9-a2ed-22b6a19df154)


Note that only the scripts with the modification with respect to the original HouseDiffusion study have been added to this repository since the corresponding study linked to this repository is built upon the HouseDiffusion. 


To use Oriented-HouseDiffusion, it is first required to clone and install [HouseDiffusion](https://github.com/aminshabani/house_diffusion) using the related repository.

Next, the following files need to be replaced with the files with the same name in the house_diffusion/house_diffusion/ directory:

- transformer.py
- rplanhg_datasets.py
- script_util.py

Next, the image_sample.py file in the house_diffusion/scripts/ directory should be removed and the following files need to be added (in the same directory):

- image_sample_FID.py
- image_sample_svg.py
- fid_score.py
  

The rest of the process, including training and testing the model is similar to the original HouseDiffusion workflow.
