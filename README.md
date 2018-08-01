# DeepSTORM
Deepl learning for Super-resolution STORM images

## Dataset
The simulated SMLM datasets from Thunderstorm are used to train CNN. The orginal diffractive images are stacked as TIFF format, and the respective .csv files record the position information of each .tiff. In the prepare, the postion and orginal image are saved as .mat.

## Framework
In the pre-processing, image_resize, image_random_crop and normalization are applied. The input size is n*(crop_size,crop_size), n represents the magnification. The label image can be acquired from xy coordinate with scipy.sparse.coo_matrix. AS the backbone of our architecture, Dense-Unet is built.

## Save
The event.log files are stored in ./log and the ckpt in ./model.
