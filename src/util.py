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

def find_centroid(vertices):
    return vertices.mean(axis=0)

def find_bbox(vertices):
    return np.amin(vertices, axis=0), np.amax(vertices, axis=0)

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

    def resolve_nested(self, other):
        vertex = self.vertices[0]
        intersections = 0
        min_trans = np.inf
        ray = np.array([0, 1])
        for q0, q1 in zip(other.vertices, other.vertices[1:] + \
            other.vertices[0]):
            point = ray_line_intersection_point(vertex, ray, q0, q1)
            if point is not None:
                if np.isclose(point, q0).all():
                    if q1[1] < point[1]:
                        translation = min(max_trans, point[1] - vertex[1])
                        intersections += 1
                elif np.isclose(point, q1).all():
                    if q0[1] < point[1]:
                        translation = min(max_trans, point[1] - vertex[1])
                        intersections += 1
                else:
                    translation = min(max_trans, point[1] - vertex[1])
                    intersections += 1

        if total % 2 == 0:
            return 0
        else:
            return intersections

    def resolve_overlap(self, other, intersections):
        max_trans = 0
        for i, j in intersections:
            p0 = self.vertices[i],
            p1 = self.vertices[i + 1 % len(self.vertices)]
            q0 = other.vertices[j],
            q1 = other.vertices[j + 1 % len(other.vertices)]

            if np.sign(p0[0] - q0[0]) != np.sign(p0[0] - q1[0]):
                y_coord = (p0[0] - q0[0]) * (q1[1] - q0[1]) / (q1[0] - q0[0])
                max_trans = max(max_trans, y_coord - p0[1])
            if np.sign(p1[0] - q0[0]) != np.sign(p1[0] - q1[0]):
                y_coord = (p1[0] - q0[0]) * (q1[1] - q0[1]) / (q1[0] - q0[0])
                max_trans = max(max_trans, y_coord - p1[1])
            if np.sign(q0[0] - p0[0]) != np.sign(q0[0] - p1[0]):
                y_coord = (q0[0] - p0[0]) * (p1[1] - p0[1]) / (p1[0] - p0[0])
                max_trans = max(max_trans, y_coord - q0[1])
            if np.sign(q1[0] - p0[0]) != np.sign(q1[0] - p1[0]):
                y_coord = (q1[0] - p0[0]) * (p1[1] - p0[1]) / (p1[0] - p0[0])
                max_trans = max(max_trans, y_coord - q1[1])

        return max_trans
