# Facial Expression Retargeting from Human to Avatar Made Easy

-------------------------------------------------------------

## Dependency

OpenMesh 6.3

Keras 2.2.0

Tensorflow 1.14.0

Numpy 1.16.4

Scipy 1.2.0

pyigl

openmesh (python)

-------------------------------------------------------------

## Pre-trained Model

Due to large file limitation on github, please download the pre-trained models from GDrive: https://drive.google.com/file/d/12UQFhxCibE0DWai6sVerzzQAQGwTDpbJ/view?usp=sharing

After extracting the zip file, please copy the "model" folder to each branch, i.e., Mery_human_transfer, Chubby_human_transfer, Conan_human_transfer.

------------------------------------------------------------

## Testing

(1) Compute deformation representation feature (RIMD feature)  -> ./RIMD_Reconstruct
(2) Concate DR feature into testing data -> See example in ./Mery_human_transfer/data/people/test_data.npy
(3) Run testing phase of each model -> For example: "cd Mery_human_transfer;  python main.py -l -t -m triplemoji;"

-------------------------------------------------------------

## Help

1). The libigl library has been replaced with a new version of python binding (beta version). However the newer version does not support well for this code. Please refer to the older version in "./libigl" and compile the pyigl.so library by: "cd libigl; mkdir build; cmake ..; make python -j4;".

2). To compile and run the RIMD_Reconstruct project, please first check the profile settings in "./RIMD_Reconstruct/RIMD.pro". Specifically, please replace the include path of OpenMesh library to your local one. 

3). In each avatar filefolder, please re-compile the get_mesh_py module; For example, "cd ./Mery_human_transfer/src/; mkdir build; cd build; cmake ..; make -j4;". 


----------------------------------------------------------------

## Acknowledgement

The repository is partially built on [get_dr_py](https://github.com/QianyiWu/get_dr_py) and [DR-Learning-for-3D-Face](https://github.com/zihangJiang/DR-Learning-for-3D-Face).
