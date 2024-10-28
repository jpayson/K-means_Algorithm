# K-Means clustering implementation

# Reads in data from a csv file and runs the K-Means algorithm with a user specified
# number of clusters and iterations. Prints and plots the results.
#
# Designed to be run on a dataset of birth rates and life expectancies for various
# countries, but only the constant data name/description variables and filename prompt
# need to be edited to customize the program to a new dataset.
#
# Note 'data point' in comments is used to refer to data of the form (x, y)
# where x and y are both floats.

import csv
import numpy as np
import matplotlib.pyplot as plt

# Constant data type used to store data as a name and data point: ('name', (x, y))
DATA_FORMAT = np.dtype([('name', 'U100'), ('xy', 'float', (2,))])

# Gives an index to each of the matplotlib.colors and select matplotlib.markers
COLORS = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y', 6:'k', 7:'w'}
MARKERS = {0:'o', 1:'s', 2:'v', 3:'^', 4:'<', 5:'>'}

# Constants used to describe elements of the dataset
DATA_NAME, DATA_NAME_PLURAL = "Country", "Countries"
DATA_X_NAME, DATA_Y_NAME = "Birth Rate", "Life Expectancy"
DATA_X_DESC, DATA_Y_DESC = "Live Births per 1,000 People per Year", "Years"
DATASET = ""


# ====
# Define a function that computes the distance between two data points
def distance(i, j):
    x_i, y_i = i[0], i[1]
    x_j, y_j = j[0], j[1]

    return ((x_j - x_i) ** 2 + (y_j - y_i) ** 2) ** 0.5

# Takes in an array i of data points and a single data point j
# Returns an array of the Euclidean distances between i and j
def array_distances(i, j):
    x_i, y_i = i[:, 0],  i[:, 1]
    x_j, y_j = j[0], j[1]

    distance_list =  ((x_j - x_i) ** 2 + (y_j - y_i) ** 2) ** 0.5
    return distance_list[:, np.newaxis]

# Takes in a number k of clusters, an array of data points, and an array of k mean data points
# Returns an array of the distances between each data point and each mean data point
def calculate_distances(k, xy_values, cluster_means):
    distances = array_distances(xy_values, cluster_means[0])
    for cluster in range(1, k):
        distances = np.hstack((distances, array_distances(xy_values, cluster_means[cluster])))

    return distances

# Takes in a 2d array of distances between data points and each cluster mean
# Returns an array of the indexes of the closest cluster to each data point
def calculate_closest_clusters(distances):
    num_elements = distances.shape[0]
    closest_clusters = np.zeros(num_elements, dtype=int)

    for i in range(num_elements):
        closest_clusters[i] = np.argmin(distances[i])
            
    return closest_clusters

# Takes in a 2d array of distances and an array of the indexes of the minimum distance
# Returns a 1d array of the minimum distances
def minimum_distances(distances, index_of_minimum):
    num_elements = distances.shape[0]
    min_distances = np.empty(num_elements, float)

    for i in range(num_elements):
        min_distances[i] = distances[i][index_of_minimum[i]]

    return min_distances

# ====
# Takes the name of a csv file, returns a numpy.ndarray of the csv data
# converted into dtype=DATA_FORMAT
def csv_to_list(filename):
    
    with open(filename, 'r') as csvfile:
        
        data_reader = csv.reader(csvfile, delimiter=',')
        next(data_reader) # Discard headers
        
        data = np.empty(0, dtype=DATA_FORMAT)

        for line in data_reader:
            # Extract values from line
            name = line[0]
            x = float(line[1])
            y = float(line[2])

            # Convert to DATA_FORMAT and add to data
            datapoint = np.array([(name, (x, y))], dtype=DATA_FORMAT)
            data = np.append(data, datapoint, axis=0)

        return data


# ====
# Initialize cluster means by returning k (x,y) points randomly from xy_values
def initialize_cluster_means(k, xy_values):
    return xy_values[np.random.choice(xy_values.shape[0], k)]

# ====
# Implement the k-means algorithm, using appropriate looping
# Prints the sum of squared distances for each iteration
# Prints the number of datapoints and mean x and y values for each cluster
# Prints the names of all of the datapoints in the cluster
# Plots the cluster as a scatterplot. Supports different color/marker combinations
# for up to 30 clusters before repeating. Means are marked as yellow diamonds.
def kmeans(data, dataset_name, k, iterations):

    # Choose starting points for means and create list for storing pyplot.axis
    cluster_means = initialize_cluster_means(k, data['xy'])
    
    for i in range(iterations):
        sum_squared_distances = 0
        
        # Calculate 2d array of distances between data['xy'] and each cluster mean
        data_distances = calculate_distances(k, data['xy'], cluster_means)

        # Calculate 1d array of the indexes of the closest cluster_means based on data_distances
        closest_clusters = calculate_closest_clusters(data_distances)
        data_distances = minimum_distances(data_distances, closest_clusters)

        # List of lists for each cluster containing the indexes of data belonging to that cluster
        cluster_data_indexes = []

        cluster_x = np.empty(0, float)
        cluster_y = np.empty(0, float)

        for cluster_num in range(k):
            cluster_data_indexes.append(np.array(np.where(closest_clusters == cluster_num))[0])

            # Strip out x and y values for this cluster only, convert to DATA_TYPE and add to clusters
            cluster_x = data['xy'][:, 0][cluster_data_indexes[cluster_num]]
            cluster_y = data['xy'][:, 1][cluster_data_indexes[cluster_num]]
            
            # Calculate new cluster mean
            cluster_x_mean = np.mean(cluster_x)
            cluster_y_mean = np.mean(cluster_y)
            cluster_means[cluster_num] = (cluster_x_mean, cluster_y_mean)

            # Calculate sum of squared distances for the cluster, add to running total
            cluster_distances_squared = data_distances[cluster_data_indexes[cluster_num]] ** 2
            sum_squared_distances += cluster_distances_squared.mean()

        print("\nIteration " + str(i + 1) + ": Sum of Squared Distances = {:1.2f}".format(sum_squared_distances))                                                         

    cluster_data_indexes = np.array(cluster_data_indexes)
    
    # Obtain indexes of x_means in ascending order
    x_mean_ascending_order = np.argsort(cluster_means, axis=0)[:, 0]
    
    # Re-order the cluster_means and cluster_data_indexes (left to right along the x axis as index increases)
    cluster_means = np.take(cluster_means, x_mean_ascending_order, axis =0)
    cluster_data_indexes = np.take(cluster_data_indexes, x_mean_ascending_order, axis =0)


    # Final Analysis and Reporting
    for cluster_num in range(k):
        # Print Results
        cluster_num_string = "Cluster " + str(cluster_num + 1)
        cluster_size = len(cluster_data_indexes[cluster_num])
        print("\n" + cluster_num_string + ": " + str(cluster_size) + " " + DATA_NAME_PLURAL + "\n")
        
        x_mean_string = "{:1.2f}".format(cluster_means[cluster_num][0])
        print("Mean " + DATA_X_NAME + " = " + x_mean_string)
        
        y_mean_string = "{:1.2f}".format(cluster_means[cluster_num][1])
        print("Mean " + DATA_Y_NAME + " = " + y_mean_string + "\n")
        
        print(DATA_NAME_PLURAL + " =", data['name'][cluster_data_indexes[cluster_num]])

        # Plot cluster, using each matplotlib color in sequence (except black and white)
        cluster_x = data['xy'][:, 0][cluster_data_indexes[cluster_num]]
        cluster_y = data['xy'][:, 1][cluster_data_indexes[cluster_num]]
        
        # Create label and plot cluster x and y values
        x_mean_string = "{:1.2f}".format(cluster_means[cluster_num][0])        # x.xx
        y_mean_string = "{:1.2f}".format(cluster_means[cluster_num][1])        # y.yy
        cluster_mean_string = "(" + x_mean_string + ", " + y_mean_string + ")" # (x.xx, y.yy)
        cluster_label = "Cluster " + str(cluster_num + 1) + ": μ = " + cluster_mean_string
        plt.scatter(cluster_x, cluster_y,
                    c = COLORS[cluster_num % 5],
                    marker = MARKERS[(cluster_num // 5) % 6],
                    edgecolors= 'k',
                    label = cluster_label)
    
    # Plot cluster means as yellow diamonds
    cluster_means_x = cluster_means[:, 0]
    cluster_means_y = cluster_means[:, 1]
    plt.scatter(cluster_means_x, cluster_means_y, c='y', edgecolors='k', marker='D')

    # Add final labels and show
    plt.title(DATA_X_NAME + " vs. " + DATA_Y_NAME + " for each " + DATA_NAME + " in " + dataset_name)
    plt.xlabel(DATA_X_NAME + " (" + DATA_X_DESC + ")")
    plt.ylabel(DATA_Y_NAME + " (" + DATA_Y_DESC + ")")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    plt.show()                               

def main():
    # Prompt user for the dataset to be used and reads in data from the appropriate file
    dataset_name = input("Which dataset should be used? Please enter 1953, 2008, or both: ")
    while dataset_name.lower()[:4] not in ["1953", "2008", "both"]:
        dataset_name = input("\nYour choice was not understood. Please enter \"1953\", \"2008\", or \"both\": ")
        
    data = csv_to_list("data"+dataset_name[:4].title()+".csv")
    if dataset_name == "both":
        dataset_name = "both 1953 and 2008"
    # Prompt user for the number of clusters
    clusters = 0
    prompt = "\nHow many clusters should be formed? Please enter a positive integer: "
    while clusters < 1:
        try:
            clusters = int(input(prompt))
            if clusters < 1:
                prompt = "\nYour choice was not understood. Please enter a whole number greater than zero: "
        except ValueError:
            prompt = "\nYour choice was not understood. Please enter a whole number greater than zero: "

    # Prompt user for the number of iterations
    iterations = 0
    prompt = "\nHow many iterations should be run? Please enter a positive integer: "
    while iterations < 1:
        try:
            iterations = int(input(prompt))
            if iterations < 1:
                prompt = "\nYour choice was not understood. Please enter a whole number greater than zero: "
        except ValueError:
            prompt = "\nYour choice was not understood. Please enter a whole number greater than zero: "
            
    kmeans(data, dataset_name, clusters, iterations)

main()
