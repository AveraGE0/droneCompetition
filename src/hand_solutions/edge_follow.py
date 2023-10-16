import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from src.hand_solutions.pixel_ordering import order_pixels

def add(x, y):
    return x+y


def get_neighbors(point, radius):
    neighbors = []
    for i_y in range(-radius, radius+1):
        for i_x in range(-radius, radius+1):
            if i_x !=0 and i_y != 0:
                neighbors += [tuple(map(add, point ,(i_x, -i_y)))]
    return neighbors


def reduce_corner_candidates(image) -> list:
    """Function to get corner candidates

    Args:
        image (_type_): image (not segemented!)

    Returns:
        list: list of indices of potential corner candidates
    """



def follow_edge(segmented_image) -> list:
    """Function to get the path on an segmented image by line following

    Args:
        segmented_image (_type_): _description_

    Returns:
        list: _description_
    """
    image = np.array(segmented_image)
    initial_point = (200, 200)
    initial_edge_point = None
    
    # Find initial edge 
    for radius in range(1, 10):
        point_right = tuple(map(add, initial_point, (radius, 0)))
        point_left = tuple(map(add, initial_point, (-radius, 0)))
        point_up = tuple(map(add, initial_point, (0, radius)))
        point_down = tuple(map(add, initial_point, (0, -radius)))
        
        if image[*point_right]:
            initial_edge_point = tuple(map(add, point_right, (radius-1, 0)))
            break
        elif image[*point_left]:
            initial_edge_point = tuple(map(add, point_right, (radius+1, 0)))
            break
        elif image[*point_up]:
            initial_edge_point = tuple(map(add, point_right, (0, radius-1)))
            break
        elif image[*point_down]:
            initial_edge_point = tuple(map(add, point_right, (0, radius+1)))
            break
    if not initial_edge_point:
        raise ValueError("No initial point found")
    print(f"Found point {initial_edge_point}")
    #while 
    for ng in get_neighbors(initial_edge_point, radius=1):
        print(ng)
        if image[*ng]:
            next_point = ng
    return [initial_edge_point, next_point]


def convert_ind_to_point(index, image_size) -> tuple:
    x = index // image_size
    y = index % image_size
    return x, y


def gpt_enhanced_method(segmented_image, im_size=400):
    #binary_image = np.array(segmented_image)
    # Perform skeletonization to extract the centerline
    _, bin_image = cv.threshold(segmented_image, 128, 255, cv.THRESH_BINARY)
    skeleton = cv.ximgproc.thinning(bin_image)
    x, y = np.where(skeleton == 255)
    points_skeleton = np.ndarray(shape=(0, 2), dtype=np.int64)
    for i in range(len(x)):
        points_skeleton = np.append(points_skeleton, [np.array((x[i], y[i]))], axis=0)
    pixel_ordered = order_pixels(points_skeleton, 400)
    return pixel_ordered


if __name__ == '__main__':
    #track = Image.open('tracks/test_track.bmp')
    track = cv.imread('tracks/test_track.bmp', cv.IMREAD_GRAYSCALE)
    skeleton = gpt_enhanced_method(track)
    #indices_skeleton = np.where(skeleton == 255)
    track = np.array(track)
    for i, index in enumerate(skeleton):
        track[*index] = i +1
    # merge images
    plt.imshow(track)
    plt.colorbar()
    #plt.imshow(skeleton, cmap='ocean')
    plt.show()

    waypoints = follow_edge(track)
    image_np = np.array(track, dtype=np.int32)
    for waypoint in waypoints:
        image_np[*waypoint] = 2
    plt.imshow(image_np)
    plt.colorbar()
    plt.show()