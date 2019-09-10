import numpy as np

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def detect_overlap(self, other):
        pass

    def is_nested(self, other):
        pass

    def resolve_overlap(self, intersections):
        pass

    def resolve_nesting(self, other):
        pass

def find_centroid(vertices):
    return vertices.mean(axis=0)

def find_bbox(vertices):
    return np.amin(vertices, axis=0), np.amax(vertices, axis=0)

def get_rotations(vertices, rotations):
    centroid = find_centroid(vertices)
    relative = vertices - centroid
    radii = np.linalg.norm(relative, axis=1)
    angles = np.arctan2(relative[..., 1], relative[..., 0])

    d_theta = 2 * np.pi / rotations
    rotations = []
    for increment in np.arange(0, 2 * np.pi, d_theta):
        angles += increment
        rotated_x = radii * np.cos(angles)
        rotated_y = radii * np.sin(angles)
        rotated_vertices = np.concatenate(rotated_x, rotated_y, axis=1)
        mins, _ = find_bbox(rotated_vertices)
        rotated_vertices -= mins
        rotations.append(Polygon(rotated_vertices))

    return rotations
