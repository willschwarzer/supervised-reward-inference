import os
import numpy as np

planes_folder = 'processed_planes'
block_types = ['negative', 'neutral', 'positive']

def line_segments(x):
    for i in range(len(x)-1):
        yield x[i], x[i+1]

def closest_point_on_segment(line_start_og, line_end_og, x_og):
    line_start = line_start_og - line_start_og
    line_end = line_end_og - line_start_og
    x = x_og - line_start_og

    norm_vec = line_end / np.linalg.norm(line_end)
    t = np.dot(norm_vec, x)
    if t < 0:
        return line_start_og
    elif t > np.linalg.norm(line_end):
        return line_end_og
    else:
        return t * norm_vec + line_start_og

def num_to_str(x):
    if x < 0:
        return 'negative'
    elif x == 0:
        return 'neutral'
    else:
        return 'positive'

def gt_to_str(gt):
    assert(len(gt) == 3)
    return '_'.join([num_to_str(x) for x in gt])

class Fetcher:
    def __init__(self):
        self.map = {}

    def comparison_point(self, point, gt, mode='gt'):
        assert(len(gt) == 3)
        key_str = gt_to_str(gt)
        if key_str not in self.map:
            return gt

        if mode == 'gt':
            return gt
        elif mode == 'centroid':
            return self.map[key_str]['centroid']
        elif mode == 'arp':
            data = self.map[key_str]
            planes = data['planes']
            hull = data['hull']
            if (np.matmul(planes, point) >= 0).all():
                return point
            else:
                best_point = None
                min_distance = float('infinity')
                for line_begin, line_end in line_segments(hull):
                    closest_point = closest_point_on_segment(line_begin, line_end, point)
                    distance = np.linalg.norm(closest_point - point)
                    if distance < min_distance:
                        min_distance = distance
                        best_point = closest_point
                return best_point


def all_block_types():
    for b1 in block_types:
        for b2 in block_types:
            for b3 in block_types:
                yield b1, b2, b3

def load_environment(envname):
    env_folder = planes_folder + os.path.sep + envname
    fetcher = Fetcher()
    for b1, b2, b3 in all_block_types():
        basename = '_'.join([b1, b2, b3])
        filename = env_folder + os.path.sep + basename + '.npy'
        if not os.path.exists(filename):
            continue
        data = np.load(filename, allow_pickle=True).item()
        fetcher.map[basename] = data
    return fetcher
