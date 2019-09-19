import numpy as np

def line_line_intersection(p0, p1, q0, q1):
    r = p1 - p0
    s = q1 - q0
    if np.isclose(np.cross(r, s), 0)[0]:
        return False
    else:
        t = (np.cross((q0 - p0), s) / np.cross(r, s))[0]
        u = (np.cross((q0 - p0), r) / np.cross(r, s))[0]
        return 0 <= t <= 1 and 0 <= u <= 1

def ray_line_intersection_point(p0, r, q0, q1):
    s = q1 - q0
    if np.isclose(np.cross(r, s), 0)[0]:
        return None
    else:
        t = (np.cross((q0 - p0), s) / np.cross(r, s))[0]
        u = (np.cross((q0 - p0), r) / np.cross(r, s))[0]
        if 0 <= t and 0 <= u <= 1:
            return np.array([p0 + r * t, q0 + s * u])

class Shape:
    def __init__(self, vertices):
        self.vertices = vertices
        self.centroid = find_centroid(shape.vertices)
        relative = self.vertices - self.centroid
        self.radii = np.linalg.norm(relative, axis=1)
        self.angles = np.arctan2(relative[..., 1], relative[..., 0])

    def get_rotation(self, angle):
        rotated_angles = self.angles + angle
        rotated_x = self.radii * np.cos(self.angles) + self.centroid
        rotated_y = self.radii * np.sin(self.angles) + self.centroid
        rotated_vertices = np.concatenate(rotated_x, rotated_y, axis=1)
        mins, _ = find_bbox(rotated_vertices)
        rotated_vertices -= mins
        return Polygon(self, self.centroid, rotated_vertices)

class Polygon:
    def __init__(self, shape, centroid, vertices):
        self.shape = shape
        self.centroid = centroid
        self.vertices = vertices

    def translate(self, new_centroid):
        self.vertices += new_centroid - self.centroid
        self.centroid = new_centroid

    def detect_intersections(self, other):
        intersections = []
        for i, (p0, p1) in enumerate(zip(self.vertices, self.vertices[1:] + \
            [self.vertices[0]])):
            for j, (q0, q1) in enumerate(zip(other.vertices,
                other.vertices[1:] + [other.vertices[0]])):
                if line_line_intersection(p0, p1, q0, q1):
                    intersections.append(i, j)

        return intersections

    def is_nested(self, other):
        vertex = self.vertices[0]
        total = 0
        ray = np.array([1, 0])
        for q0, q1 in zip(other.vertices, other.vertices[1:] + \
            other.vertices[0]):
            point = ray_line_intersection_point(vertex, ray, q0, q1)
            if point is not None:
                if np.isclose(point, q0).all():
                    if q1[1] < point[1]:
                        total += 1
                elif np.isclose(point, q1).all():
                    if q0[1] < point[1]:
                        total += 1
                else:
                    total += 1

        return total % 2 == 1

    def resolve_overlap(self, other, intersections):
        pass

    def resolve_nesting(self, other):
        pass

def find_centroid(vertices):
    return vertices.mean(axis=0)

def find_bbox(vertices):
    return np.amin(vertices, axis=0), np.amax(vertices, axis=0)
