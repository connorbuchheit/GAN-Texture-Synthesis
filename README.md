# GAN-Texture-Synthesis

This is our final project for CS2831. We aim to implement a rudimentary version of the algorithm from "Learning Texture Manifolds with the Periodic Spatial GAN” by Bergmann et al., 2017, and to extend the implementation to include additional parameters \alpha that can accommodate perspective distortion that is caused when a planar texture sample is observed from a non-frontal view.

Paths to take: Can apply a homography in cv2

How to run:
The notebook called `notebookParatraining.ipynb` is all that is necessary to run the GAN on the specified image. If you want to change the image type generated, you can do so in the data pull cell by using the image names from `dtd_folder/dtd`
