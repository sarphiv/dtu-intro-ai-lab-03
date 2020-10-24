import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import numpy as np
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def rescale_image(image, max_length: int):
    options = {
        "mode": "reflect",
        "multichannel": True, 
        "anti_aliasing": True
    }

    #Scale image based on maximum side length
    scale_ratio = max_length / max(image.shape[0], image.shape[1])

    return rescale(image, scale_ratio, **options)



def image_to_data_point(image):
    """
    Serializes list of list of (2D) row pixel colors to list of (1D) pixel colors.
    Also transforms the data point values.
    """

    lab_image = rgb2lab(image)
    one_d_image = lab_image.reshape(-1, 3)

    #TODO: We can include position in the serialization step here


    return one_d_image


def data_point_to_image(data_points, original_shape):
    """
    Deserializes list of (1D) pixel colors to list of list of (2D) row pixel colors.
    Also "untransfroms" the data point values.
    """

    #TODO: If adding position, remember to remove position serialization step here

    lab_image = data_points.reshape(*original_shape)
    image = lab2rgb(lab_image)


    return image



def calculate_clusters(data_points, num_clusters: int):
    """
    Returns (cluster_assignments, cluster_centers)
    """

    def cluster_assignments(data_points, centers):
        #Calculate distances from each point to centers
        #For each data point, get index of the smallest distance.
        return np.argmin(euclidean_distances(data_points, centers), axis=1)

    def cluster_centers(data_points, assignments, prev_centers):
        centers = prev_centers.copy()

        for cluster_id, prev_center in enumerate(prev_centers):
            cluster_points = data_points[assignments == cluster_id]

            if len(cluster_points):
                centers[cluster_id] = cluster_points.mean(0)

        return centers


    #Initialize centers
    color_channel_mean = data_points.mean(0)
    color_channel_noise = np.random.randn(3) / 10

    centers = np.array([color_channel_mean + color_channel_noise for _ in range(num_clusters)])
    assignments = None

    #Assign clusters, update centers, repeat until converged
    while True:
        #NOTE: Using shallow copy as we never mutate it
        prev_centers = centers

        assignments = cluster_assignments(data_points, centers)
        centers = cluster_centers(data_points, assignments, centers)

        if np.array_equal(centers, prev_centers):
            break


    return (assignments, centers)


def centers_as_data_points(assignments, centers):
    #Each data point has an assignment in the same index, 
    # replace the assignment with its equivalent center
    return centers[assignments, :]




#Load image
image_raw = imread('images/soccerball.jpg')

#---Random note: would it make sense to not scale the image if we trie to compare a compressed with a non compressed?

#Scale image
image_max_length = 166
image = rescale_image(image_raw, image_max_length)

#Convert to format K-means works on
data_points = image_to_data_point(image)

#--Note: Would it be possible to add a method that makes the num-clusters assignment more dynamic?
#-- somehow make the amount of clusters used based on the amount of contrasting or very diffrent colours

#Run K-means
num_clusters = 3
clusters = calculate_clusters(data_points, num_clusters)

#Generate segmented data
segmented_data_points = centers_as_data_points(*clusters)

#Unconvert to image form again
segmented_image = data_point_to_image(segmented_data_points, image.shape)


#Display image
plt.figure()
plt.imshow(segmented_image)
plt.axis('off')
plt.show()
