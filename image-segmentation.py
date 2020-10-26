from os import replace
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import numpy as np
import math
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt



max_image_side_length = -1         #Use the value -1, for no resizing
num_cluster_factor = 16
max_k_means_iterations = 30
max_center_similarity = 1



def rescale_image(image, max_length: int):
    """
    Also transforms RGB values to a standard format
    """
    options = {
        "mode": "reflect",
        "multichannel": True, 
        "anti_aliasing": True
    }

    #Scale image based on maximum side length
    scale_ratio = max_length / max(image.shape[0], image.shape[1])

    return rescale(image, scale_ratio, **options)



def image_to_k_means_format(image):
    """
    Serializes list of list of (2D) row pixel colors to list of (1D) pixel colors.
    Also transforms the pixel values to work better with euclidean distances.
    """
    lab_image = rgb2lab(image)
    one_d_image = lab_image.reshape(-1, 3)


    return one_d_image


def clusters_to_file_format(assignments, centers):
    """
    Store assignments as bytes.
    Convert centers from LAB to RGB colors and store as bytes
    """
    return (
        
        assignments.astype(np.uint8),
        (lab2rgb([centers]) * 255).astype(np.uint8).reshape(-1, 3)
    )


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
    #WARN: This leads to non-deterministic compression
    color_channel_noise = lambda: np.random.randn(3) / 100

    centers = np.array([color_channel_mean + color_channel_noise() for _ in range(num_clusters)])
    assignments = None

    #Print progress
    progress_bars = 30
    current_progress = 0
    progress_bar_title = "Cluster analysis progress "
    print(progress_bar_title + ' ' * (progress_bars - len(progress_bar_title)) + '|')

    #Assign clusters, update centers, repeat max iterations times or until converged
    for i in range(max_k_means_iterations):
        #NOTE: Using shallow copy as we never mutate it
        prev_centers = centers

        assignments = cluster_assignments(data_points, centers)
        centers = cluster_centers(data_points, assignments, centers)

        #If next progress bar should be printed, print it
        bar_progress = math.floor(i / max_k_means_iterations * progress_bars)
        if  bar_progress > current_progress:
            print('-', end='')
            current_progress = bar_progress

        #If converged, break out of loop
        if np.array_equal(centers, prev_centers):
            break

    #Print new line for next commands
    print('')


    return (assignments, centers)



def remove_similar_centers(assignments, centers, max_similarity):
    #Calculate distance between all centers and mark ones too similart
    comparisons = euclidean_distances(centers, centers) < max_similarity
    #Find similar indexes and their replacements
    similar_indexes = []
    for i, comp in enumerate(comparisons):
        for j, similar in enumerate(comp[i:]):
            replacement_index = i + j
            #If similar and not comparing to itself, 
            # mark as index similar and mark replacement index
            if similar and i != replacement_index:
                similar_indexes.append((i, replacement_index))


    #Delete all first indexes that have similar replacements
    centers = np.delete(centers, [i for i, _ in similar_indexes], axis=0)

    #Replace all deleted assignments with their assigned replacement
    for i, replacement_index in similar_indexes:
        assignments[assignments == i] = replacement_index

    #Collapse all assignment indexes to start from 0 and count up by +1
    for zero_based_index, current_index in enumerate(np.unique(assignments)):
        assignments[assignments == current_index] = zero_based_index


    return (assignments, centers)



def load_image(file_name, max_side_length = -1):
    #Load image
    image_raw = imread(file_name)
    #Scale and return image
    return rescale_image(image_raw, max(image_raw.shape[0], image_raw.shape[1]) if max_side_length == -1 else max_side_length)


def compress_image(image, output_file_name):
    #Convert to format K-means works on
    color_points = image_to_k_means_format(image)

    #Run K-means
    #NOTE: Adding one to deal with edge case of a 1x1 image not getting any clusters
    #NOTE: Limiting max clusters to fit inside a byte
    num_clusters = min(1 + math.ceil(num_cluster_factor * math.log2(image.shape[0] * image.shape[1])), 255)
    clusters = calculate_clusters(color_points, num_clusters)

    #Remove similar centers and adjust assignment numbers to match new centers
    clusters = remove_similar_centers(*clusters, max_similarity=max_center_similarity)

    #Save compressed image to file
    np.savez_compressed(output_file_name, [image.shape[0], image.shape[1]], *clusters_to_file_format(*clusters))


    return clusters


def uncompress_image(file_name):
    #Load in compressed image
    [image_shape, assignments, centers] = np.load(file_name).values()

    #Replace assignment indexes with the corresponding pixel values for their centers
    image_pixels = centers[assignments, :]

    #Unconvert to image form again
    image = image_pixels.reshape(*image_shape, 3)


    return image


def calculate_loss(image_reference, image_compressed):
    reference = image_to_k_means_format(image_reference)
    compressed = image_to_k_means_format(image_compressed)

    return np.sum(np.sqrt(np.sum((compressed - reference)**2, axis=1))) / len(reference)



#Load image
#NOTE: Scaling for faster testing purposes. Disable for experiment
image = load_image("images/gates.jpg", max_image_side_length)

#Compress image to file
compress_image(image, "image.npz")

#Load and uncompress image from file
compressed_image = uncompress_image("image.npz")

#Calculate and print loss
loss = calculate_loss(image, compressed_image)
print(f"Loss:\n{loss}")


#Display image
plt.figure(0)
plt.imshow(compressed_image)
plt.axis('off')
plt.figure(1)
plt.imshow(image)
plt.axis('off')
plt.show()
