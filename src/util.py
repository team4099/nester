import numpy as np

def line_intersection(p0, p1, q0, q1):
    r = p1 - p0
    s = q1 - q0
    if np.isclose(np.cross(r, s), 0)[0]:
        return False
    else:
        t = (np.cross((q0 - p0), s) / np.cross(r, s))[0]
        u = (np.cross((q0 - p0), r) / np.cross(r, s))[0]
        return 0 <= t <= 1 and 0 <= u <= 1

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def detect_intersections(self, other):
        intersections = []
        for i, (p0, p1) in enumerate(zip(self.vertices, self.vertices[1:] + \
            [self.vertices[0]])):
            for j, (q0, q1) in enumerate(zip(other.vertices,
                other.vertices[1:] + [self.vertices[0]])):
                if line_intersection(p0, p1, q0, q1):
                    intersections.append(i, j)
        
        return intersections

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
        rotated_x = radii * np.cos(angles) + centroid
        rotated_y = radii * np.sin(angles) + centroid
        rotated_vertices = np.concatenate(rotated_x, rotated_y, axis=1)
        mins, _ = find_bbox(rotated_vertices)
        rotated_vertices -= mins
        rotations.append(Polygon(rotated_vertices))

    return rotations
