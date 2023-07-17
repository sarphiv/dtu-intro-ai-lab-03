**Lab 3 - Image compression with k-means segmentation** October 28, 2020


August Hertz Bugge & Kris Back Kruse & sarphiv (redacted)

# Pre notes
For the full rapport read report.pdf

The following is a teaser from the report.


# Abstract 

Storing images is necessary for many AI tasks, however, data storage at
large scale is costly. A widely used technique is to compress images
using lossy algorithms like JPEG, but even a small improvement can save
money. We compare k-means clustering as a lossy compression algorithm
vs. JPEG on 1890 images using qualitatively chosen parameters for our
k-means clustering. Our algorithm has a compression ratio of
35% better than JPEG. This indicates high compression ratios
can be obtained if parameters are optimized for a given use case.

# Introduction 

Storing large amounts of images requires a lot of storage space which is
costly. Many algorithms exist, and is used in different situations, to
compressed images. We want to compare the compression of an image, using
k-means clustering and the widely used "JPEG" algorithm, in terms of
space gained. Our hypothesis is that, using the k-means clustering
algorithm, we will get within 20% points of the compression ratio of
the "JPEG" algorithm, when compared to the equivalent bitmap file.

# References 

-   Image data set: <http://images.cocodataset.org/zips/val2017.zip>

-   Source code: <https://github.com/sarphiv/dtu-intro-ai-lab-03>

# Learning outcome 

Processing many images requires a lot of computational power. In the
future, more time should be spent error-checking, optimizing, and
preparing as this could ironically lead to a loss of time.
