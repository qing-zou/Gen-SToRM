# Gen-SToRM
The source code for Dynamic imaging using a deep generative SToRM (Gen-SToRM) model

## Reference papers
Dynamic imaging using a deep generative SToRM (Gen-SToRM) model by Q. Zou, A. Ahmed, P. Nagpal, S. Kruger and M. Jacob in IEEE Transactions on Medical Imaging

Link: https://arxiv.org/pdf/2102.00034.pdf


#### What this code do:
In the above paper, we propose a generative SToRM model for the reconstruction of free-breathing and ungated cardiac MRI. We assume that the images in the time series are some non-linear map of the points that are lying in the low dimensional latent space. We use a CNN to build the non-linear map as the structure of CNN can implicitly offer spitial regularization [1]. We also add two regularization terms to further constrain the solution: the network regularization and the latent vector regularization. In the paper, we also propose a progressive training-in-time approach to speed up the reconstruction process. Note that this proposed scheme needs only the highly undersampled k-t space data. So this scheme is unsupervised, meaning that no fully sampled training data are needed.

The code then did the jobs that are mentioned in the above paragraph.

#### The idea of the work and the cost function:
The following figure shows the basic idea of the work.
![image](https://user-images.githubusercontent.com/36931917/114895176-e4ab6f00-9dd4-11eb-91c7-5dc3ad214f8b.png)

We feed a set of latent vectors into the generator (G_theta) and the generator is then able to generate the images in the time series. We then perform the non-uniform Fourier Transform (NUFFT) to get the k-t space data of the generated images and compare them with the collected k-t space data. Based on which, we have the following cost function and the code solves for the cost function:
![image](https://user-images.githubusercontent.com/36931917/114894290-20920480-9dd4-11eb-93ce-58ef3e4a33ed.png)

A_i here means the non-uniform Fourier transform. b is the acquired undersampled k-t space data. In this work, we use the golden angle spiral trajectories to acquired the k-t space data.

#### The progressive training-in-time approach
Since the data in this work are collected using the spiral trajectories and each frame will have different spirals and hence we need different NUFFT operators for different frames. Therefore the computational complexity will be very high to directly solving the cost function. Hence we propose a progressive training-in-time approach to speed up the process. The following figure shows the idea of this approach. The detailed description of this approch can be found in the paper above.
![image](https://user-images.githubusercontent.com/36931917/114896036-99459080-9dd5-11eb-89c1-cec952c4015f.png)

#### Main benefits of Gen-SToRM:
1. Unlike the traditional CNN based approach, Gen-SToRM does not need extensive fully sampled training data.
2. Gen-SToRM is able to estimate the phases directly from the undersampled k-t space data and hence no navigator is needed.
3. The memory footprint of Gen-SToRM is dependent on the number of network paprameters and the very low dimensional latenet vector and we do not need to store all the images in the time series comparing to the traditional manifold methods for free-breathing and ungated cardiac MRI.

#### Dependencies:
The code relies on the NUFFT operator and we used the torchkbnufft repository to have the NUFFT operator. But note that the torchkbnufft version used in this repository is not the most recent one. You can download the .zip file for the torchkbnufft under this repository. To use it, you can just unzip it and put it in your working folder.
Note that the torchkbnufft version that we used in this repository does not support complex number. However, the most recent version (https://github.com/mmuckley/torchkbnufft) is able to support complex number. But you need to make sure that your PyTorch version is above 1.5.0 so that complex type data can be supported.

Main file: gen_storm_main.py
