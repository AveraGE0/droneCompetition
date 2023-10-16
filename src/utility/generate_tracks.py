import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle
from math import sqrt


def generate_track(way_points: list[tuple], resolution: int, track_width=10) -> np.ndarray:
    """_summary_

    Args:
        way_points (list[tuple]): list of points (x, y tuples) that the track follows
        resolution (int): amount of pixels in x and y direction
        track_width (int, optional): width of a line

    Returns:
        np.ndarray: resulting image with tracks as 1 and background as 0
    """
    previous_point = None
    track = Image.new('1', (resolution, resolution))
    for point in way_points:
        if not previous_point:
            previous_point = point
            continue
        # make direct line
        draw = ImageDraw.Draw(track) 
        draw.line([previous_point, point], fill=1, width=track_width)
        #track.show()
        previous_point = point
    return np.array(track, dtype=np.int8)


if __name__ == '__main__':
    test_tracks = {
        "tt1": [(500, 50), (500, 900)],
        "tt2": [(300, 60), (500, 500), (700, 600), (900, 900)],
        "tt3": [(500, 50), (500, 600), (600, 200), (600, 700), (300, 750), (100, 900)],
        #"tt4": [],
        #"tt5": []
    }
    start_points = []
    end_areas = []
    maps_shape = (1000, 1000, len(test_tracks))
    tracks = np.zeros(shape=(1000, 1000, len(test_tracks)))
    for i, (name, test_track) in enumerate(test_tracks.items()):
        track = generate_track(test_track, 1000)
        # make start point with constant offset
        offset_value = 3
        diff_x, diff_y = (test_track[1][0] - test_track[0][0]), (test_track[1][1] - test_track[0][1])
        norm = sqrt(diff_x**2 + diff_y**2)
        start_points.append((test_track[0][0] + int((diff_x/norm)*3), test_track[0][1] + int((diff_y/norm)*3)))
        
        end_point_x, end_point_y = test_track[-1]
        end_areas.append((end_point_x - 10, end_point_y - 10, end_point_x + 10, end_point_y + 10))
        # vis end area and start point, row, col, therefore index with y, x
        #track[end_areas[-1][1]:end_areas[-1][3], end_areas[-1][0]:end_areas[-1][2]] = 2
        #track[start_points[-1][1] + 5, start_points[-1][0]] = 3
        #break   # just to debug
        tracks[:, :, i] = track
    #plt.imshow(tracks[:,:,0])
    #plt.show()
    np.save(Path('tracks/test_tracks'), tracks)
    with open('tracks/test_tracks_info.pickle', "wb") as out_file:
        pickle.dump({"start_points":start_points, "end_areas": end_areas, "maps_shape": maps_shape}, out_file)