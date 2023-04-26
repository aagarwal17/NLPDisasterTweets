Arun Agarwal

Professor Abha Belorkar

CIS 4496 - Projects in Data Science, Honors Contract

April 28th, 2023

# Data Report

**Data Introduction:**

Our competition data primarily consists of 256x256 RGB images of two groups: (1) Monet paintings and (2) camera photos.
Specifically, our initial dataset as provided by Kaggle includes 300 Monet paintings and 7038 photos in both JPEG and TFRecord format (TensorFlow's custom data format).
We will primarily work with the TFRecord format; however, we will have the JPEG files as a fallback should any difficulties arise when using the TFRecord format.
The total competition data including both formats amounts to 385.87 MB---small enough to be stored on a personal computer.
Our group also utilizes image data for additional paintings provided by the authors for not only Monet but Van Gogh, Ukiyo-e, and Cezanne as well.

While the data size is relatively small, the problem requires intense computation within a limited timeframe, so we make the data accessible by TPU-equipped machines for the best results.
Kaggle conveniently provides access to the initial dataset stored in their systems through their Kaggle Notebooks (Jupyter Notebook service provided by Kaggle) which provide free limited access to TPU computers.
Indeed, we use Kaggle Notebooks to operate on the data and perform the majority of the computation required for the project.
So far in the project, we have not run into rate limitation issues on Kaggle, but we have run into a variety of problems involving the use of the Temple HPC node, such as the other users hogging the GPUs on the node, along with issues finding access to any useable GPU on the Node.
Due to the relative reliance on Kaggle compared with the Temple HPC node, we anticipate working on Kaggle for the remainder of the project.

**Data Cleaning and Processing Techniques:**

For this project, we needed to generate multiple models, as our scope extends beyond the Kaggle competition to include more generic models using the training and test datasets for Monet, Van Gogh, Cezanne, and Ukiyo-e, provided by the authors.
Thus, we discuss data augmentation with regard to the competition and the other artists separately.

With respect to the Kaggle Competition, we used approximately 300 Monet paintings and 7,000 photos that were pulled from the competition.
These paintings and photos were pre-converted to TFREC format by the competition authors.
Therefore, when we initially developed our model, our model took in TFREC data as its input.
TFREC (short for TensorFlow record) is a custom data format to store sequences of binary records (6).
With the TFREC format, we can make use of the TensorFlow tf.data API to easily and efficiently read and process the image data (7).
For example, we used TFREC-related objects in Tensorflow, such as the TFRecordDataset object and its various methods, such as .batch(), or .shuffle(), to shuffle our data and get batches of our data easily.
TFRecordDataset also includes a .map() method with a "num_parallel_calls" option, which allows us to read and augment the data in parallel via multithreaded processing.
Setting "num_parallel_calls" to tf.data.AUTOTUNE, the API will automatically determine how many parallel calls can be made depending on CPU availability (7).

Since we were working with "pre-cleaned" data that was provided by the competition, we did not have to make use of any data-cleaning techniques to make our data usable.
We were also uncomfortable with the idea of filtering out certain pieces of data, as we already were operating with a relatively small dataset.

However, we did experiment with a wide variety of augmentation for our data in the competition.
Initially, our group experimented with Differentiable Augmentation (DiffAugment)---a simple method that improves the data efficiency of GANs by imposing various types of differentiable augmentations (e.g. color, translation, and "cutout" augmentations) on both the real and generated images (2).
Unfortunately, this technique did not serve to benefit our performance, as many of the augmentations that were applied to our paintings seemed to alter the artist's distinctive painting style, such as by altering the paintings' colors poorly.
We also ran into some time and memory limits when trying to implement the technique.
The research paper that introduced the DiffAugment technique discusses that it should be used for "vanilla" GANs instead of GANs used for image style transfer.
Instead of using DiffAugment which augments the images in each training step, we have opted to augment all the images, including Monet paintings and photos, through randomly resizing, cropping, rotating (by a multiple of 90 degrees), and flipping the images before training even begins.

![](../Project/Images/image_augmentation.png)

Figure 1: Examples of augmented images using resizing, cropping, rotation, and flipping

(self-generated)

To verify that these augmentations are working as expected, we have created a script to save the augmented images, as shown in Figure 1.
These augmentations boosted the performance of our model, as they introduced more diversity into the training data without compromising the artist's distinctive style.
We discuss the performance improvement further in the Model and Performance Report.
More effective augmentation techniques for images that do not disrupt a distinctive style of an image is an area of research that should be explored further in the near future.

With respect to our generic models, we did not explicitly use the competition dataset, instead we used the datasets that were provided by the CycleGAN authors.
For each of the four artists, the authors provided four folders: training data for paintings, training data for photographs, testing data for paintings, and testing data for photos.
Unlike the data in the competition, the author's data were in JPG format.
Therefore, we needed to convert these JPG images into TFREC format to be usable in our pipeline.
This required writing multiple scripts to implement the process.
In addition, we developed a standardized train/test split based on the CycleGAN's authors\' data.
Thus, we generated the testing data for these models by augmenting 10% of the existing training data for an artist through flips and rotations.
We do not crop the images in these new augmentations in an attempt to reduce pixelation.
This data was then added to the existing test dataset for that artist, allowing us to expand the test data size while also introducing variation.

**Feature Extraction Techniques:**

In a similar vein to the data cleaning and processing techniques, we did not use any explicit feature extraction techniques from our paintings.
Since we used CycleGANs from the start of the project, we used convolutional neural networks (CNNs) within our model in order to process and gather features from the images.
One of the key features about CNNs is that they learn what lower level features to extract from an image during the training, rather than those features being manually extracted by humans.
Due to this, manual feature extraction did not occur prior to the feature being extracted by the CNNs in our model.

**Trends in Our Data:**

As a step toward better understanding our data, we have created a script to plot the RGB distribution of a set of images to understand the data better.
We compare the RGB distribution amongst images as well as between photos and paintings to determine how to modify our model.
Figure 2 shows a plot of the RGB distribution for the first 100 Monet paintings for our generic model, the first 100 photos, and the first 100 photo-to-Monet images (the zeros are excluded since they distort the plot).
Figures 3--5 display these distributions for Ukiyo-e, Cezanne, and Van Gogh, respectively.
We observe that when generating the photo-to-Monet images, the images tend toward the RGB distribution of the Monet paintings, as expected.
Therefore, these distributions serve as a small sanity check that our generic model is transferring the color/style well.
It is interesting to note that the distributions of original paintings in comparison to the converted painting distributions appear more jagged, suggesting that our models tend to smooth out the color distributions.
These figures also contain a black line for the brightness, which represents the luminance of the images.
This is calculated by the following formula: brightness = 0.2126R + 0.7152G + 0.0722B, where R, G, and B, represent the red, green, and blue distributions, respectively (5).
Viewing the brightness distribution provides an additional way of analyzing our data and validating our results.
The RGB distribution of individual images can be examined on our website, as described later.

[<img src="../Project/Images/monet_painting_rgb.png" width="256" height="256">](../Project/Images/monet_painting_rgb.png)
[<img src="../Project/Images/monet_photo_rgb.png" width="256" height="256">](../Project/Images/monet_photo_rgb.png)
[<img src="../Project/Images/monet_generated_rgb.png" width="256" height="256">](../Project/Images/monet_generated_rgb.png)

Figure 2: RGB distribution of the first 100 Monet paintings (left), the first 100 photos (middle), and the first 100 generated photo-to-Monet images with our generic model (right), with zeros excluded (3)

[<img src="../Project/Images/ukiyoe_painting_rgb.png" width="256" height="256">](../Project/Images/ukiyoe_painting_rgb.png)
[<img src="../Project/Images/ukiyoe_photo_rgb.png" width="256" height="256">](../Project/Images/ukiyoe_photo_rgb.png)
[<img src="../Project/Images/ukiyoe_generated_rgb.png" width="256" height="256">](../Project/Images/ukiyoe_generated_rgb.png)

Figure 3: RGB Distribution of the first 100 Ukiyo-e paintings (left), the first 100 photos (middle), and the first 100 generated photo-to-Ukiyo-e images (right), with zeros excluded (3)

[<img src="../Project/Images/cezanne_painting_rgb.png" width="256" height="256">](../Project/Images/cezanne_painting_rgb.png)
[<img src="../Project/Images/cezanne_photo_rgb.png" width="256" height="256">](../Project/Images/cezanne_photo_rgb.png)
[<img src="../Project/Images/cezanne_generated_rgb.png" width="256" height="256">](../Project/Images/cezanne_generated_rgb.png)

Figure 4: RGB Distribution of the first 100 Cezanne Paintings (left), the first 100 photos (middle), and the first 100 generated photo-to-Cezanne images (right), with zeroes excluded (3)

[<img src="../Project/Images/vangogh_painting_rgb.png" width="256" height="256">](../Project/Images/vangogh_painting_rgb.png)
[<img src="../Project/Images/vangogh_photo_rgb.png" width="256" height="256">](../Project/Images/vangogh_photo_rgb.png)
[<img src="../Project/Images/vangogh_generated_rgb.png" width="256" height="256">](../Project/Images/vangogh_generated_rgb.png)

Figure 5: RGB Distribution of the first 100 Van Gogh Paintings (left), the first 100 photos (middle), and the first 100 generated photo-to-Van Gogh images (right), with zeroes excluded (3)

We would also like to mention that effort was made to create other exploratory data analysis figures/charts or at least examine the data further.
For example, we looked into analyzing brush strokes to boost model performance, as suggested by the Professor, but more time will be required before we are capable of implementing this (8).
After extensive research, we felt that any further data exploration would be too difficult without labeled images or the charts that would be created would not provide any additional insight to that being provided by the RGB distribution charts.
We also want to point out that we did not notice any other competitors exploring the data either.

**Does the data foretell any issues that may arise in later stages of
the project lifecycle?**

While the data did not explicitly tell us of any issues that arose in the project lifecycle, the nature of the data did predict some issues that we ran into later on in the project.
First and foremost, the relatively small number of paintings in the dataset, especially in the competition data, likely hindered the performance of our models.
While it is not possible to increase the amount of paintings that Monet painted, working with a small sample size likely caused our model to experience some amount of overfitting.
Like many deep learning architectures, GANs experience benefits with larger amounts of training data, so the small amount of training data likely hindered our GANs, which in turn hindered our CycleGAN models.
Anticipating this issue, our team experimented with increasing the amount of training data in order to create more "generic" models, along with augmenting our training data.

Another issue in the project lifecycle that our data foreshadowed was the relatively long runtimes for generating weights for our models.
While we had a relatively small amount of images, each image itself is dense with data, as it can be represented as a 256 X 256 X 3 tensor.
Thus, for one of our generic models with hundreds of epochs and operating with thousands of images per epoch, it could take hours or potentially days to train our models.
These long runtimes caused significant delays and were a major barrier in experimenting with our various models.
Despite there being a relatively small amount of image data, the high density of data for an image foreshadowed difficulties we had in training our models.
This is a common problem, so our team used a variety of training accelerators in our training process, such as TPUs (Tensor Processing Units), which are specifically designed to optimize the training of image data.
The use of TPUs made the training of our CycleGANs feasible for this project, but constraints still existed, as creating a top-performing generic model took approximately 8 hours on a TPU.
While technological advancements have enabled us to participate in CycleGAN modeling, advancements in AI accelerators would have lowered our barriers even further and allowed more experimentation with our models.

**References:**

1.  Jang, A., Uzsoy, A. S., & Culliton, P. (2020). _I\'m Something of a Painter Myself_. Kaggle. Retrieved April 3, 2023, from [https://www.kaggle.com/competitions/gan-getting-started](https://www.kaggle.com/competitions/gan-getting-started)

2.  Zhao, S., Liu, Z., Lin, J., Zhu, J.-Y., & Han, S. (2020, December 7). _Differentiable Augmentation for Data-Efﬁcient GAN Training_. Arxiv. Retrieved April 5, 2023, from [https://arxiv.org/pdf/2006.10738](https://arxiv.org/pdf/2006.10738)

3.  Zhu, J.-Y. (2023, March). _Junyanz/Pytorch-Cyclegan-and-pix2pix: Image-to-image translation in PyTorch_. GitHub. Retrieved April 6, 2023, from [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

4.  Wikipedia. (2023, March 31). _Luma (video)_. Wikipedia. Retrieved April 6, 2023, from [https://en.wikipedia.org/wiki/Luma\_(video)](<https://en.wikipedia.org/wiki/Luma_(video)>)

5.  Stack Overflow. (2009, February 27). _Formula to determine perceived brightness of RGB color_. Stack Overflow. Retrieved April 6, 2023, from [https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color/596243#596243](https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color/596243#596243)

6.  TensorFlow. (2022, December 15). _TFRecord and tf.train.Example_. TensorFlow. Retrieved April 6, 2023, from https://www.tensorflow.org/tutorials/load_data/tfrecord

7.  TensorFlow. (2022, December 15). _Better performance with the tf.data API_ . TensorFlow. Retrieved April 6, 2023, from [https://www.tensorflow.org/guide/data_performance](https://www.tensorflow.org/guide/data_performance)

8.  Kotovenko, D., Wright, M., Heimbrecht, A., & Ommer, B. (2021, March 31). _Rethinking Style Transfer: From Pixels to Parameterized Brushstrokes_. arXiv.org. Retrieved April 6, 2023, from https://arxiv.org/abs/2103.17185
