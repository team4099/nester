import numpy as np
from enum import Enum

TOL = 1e-5

class Intersection(Enum):
    Collinear = 0
    Oblique = 1
    Skew = 2

def line_line_intersection(p0, p1, q0, q1):
    r = p1 - p0
    s = q1 - q0
    if np.isclose(np.cross(r, s), 0).all():
        if not np.isclose(np.cross((q0 - p0), r), 0).all():
            return Intersection.Skew

        length = np.dot(r, r)
        t0 = np.dot((q0 - p0), r) / length
        t1 = t0 + np.dot(s, r) / length
        if t0 <= t1 and min(1, t1) > max(0, t0) or \
            t1 <= t0 and min(1, t0) > max(0, t1):
            return Intersection.Collinear
    else:
        t = np.cross((q0 - p0), s) / np.cross(r, s)
        u = np.cross((q0 - p0), r) / np.cross(r, s)
        if 0 <= t <= 1 and 0 <= u <= 1:
            return Intersection.Oblique

    return Intersection.Skew

def ray_line_intersection_point(p0, r, q0, q1):
    s = q1 - q0
    if not np.isclose(np.cross(r, s), 0).all():
        t = np.cross((q0 - p0), s) / np.cross(r, s)
        u = np.cross((q0 - p0), r) / np.cross(r, s)
        if 0 <= t and 0 <= u <= 1:
            return p0 + r * t

    return Intersection.Skew

def pir(p0, p1, q0, q1, intersection):
    if intersection is Intersection.Collinear:
        py0, py1 = sorted((p0[1], p1[1]))
        qy = max((q0[1], q1[1]))
        y = min(py1, qy)
        return y - py0
    else:
        p0, p1 = sorted(([*p0], [*p1]))
        q0, q1 = sorted(([*q0], [*q1]))
        max_trans = 0
        if np.isclose(q0[0], q1[0]):
            if p0[0] < q0[0] + TOL < p1[0] or p0[0] < q0[0] - TOL < p1[0]:
                y_coord = p0[1] + (q0[0] - p0[0]) * (p1[1] - p0[1]) / \
                    (p1[0] - p0[0])
                max_trans = max(max_trans, max(q0[1], q1[1]) - y_coord)
        elif np.isclose(p0[0], p1[0]):
            if q0[0] < p0[0] + TOL < q1[0] or q0[0] < p0[0] - TOL < q1[0]:
                y_coord = q0[1] + (p0[0] - q0[0]) * (q1[1] - q0[1]) / \
                    (q1[0] - q0[0])
                max_trans = max(max_trans, y_coord - min(p0[1], p1[1]))
        else:
            if q0[0] <= p0[0] + TOL <= q1[0] or q0[0] <= p0[0] - TOL <= q1[0]:
                y_coord = q0[1] + (p0[0] - q0[0]) * (q1[1] - q0[1]) / \
                    (q1[0] - q0[0])
                max_trans = max(max_trans, y_coord - p0[1])
            if q0[0] <= p1[0] + TOL <= q1[0] or q0[0] <= p1[0] - TOL <= q1[0]:
                y_coord = q0[1] + (p1[0] - q0[0]) * (q1[1] - q0[1]) / \
                    (q1[0] - q0[0])
                max_trans = max(max_trans, y_coord - p1[1])
            if p0[0] <= q0[0] + TOL <= p1[0] or p0[0] <= q0[0] - TOL <= p1[0]:
                y_coord = p0[1] + (q0[0] - p0[0]) * (p1[1] - p0[1]) / \
                    (p1[0] - p0[0])
                max_trans = max(max_trans, q0[1] - y_coord)
            if p0[0] <= q1[0] + TOL <= p1[0] or p0[0] <= q1[0] - TOL <= p1[0]:
                y_coord = p0[1] + (q1[0] - p0[0]) * (p1[1] - p0[1]) / \
                    (p1[0] - p0[0])
                max_trans = max(max_trans, q1[1] - y_coord)

        return max_trans

def find_centroid(vertices):
    return np.mean(vertices, axis=0)

def find_bbox(vertices):
    return np.amin(vertices, axis=0), np.amax(vertices, axis=0)

class Shape:
    def __init__(self, vertices):
        self.vertices = vertices
        self.centroid = find_centroid(self.vertices)
        relative = self.vertices - self.centroid
        self.radii = np.linalg.norm(relative, axis=1)
        self.angles = np.arctan2(relative[..., 1], relative[..., 0])
        self.rotations = {}

    def get_rotation(self, angle):
        if angle not in self.rotations:
            rotated_angles = self.angles + angle
            rotated_x = self.radii * np.cos(rotated_angles)
            rotated_y = self.radii * np.sin(rotated_angles)
            rotated_vertices = np.column_stack((rotated_x, rotated_y)) + \
                self.centroid
            mins, _ = find_bbox(rotated_vertices)
            rotated_vertices -= mins
            self.rotations[angle] = (self.centroid - mins, rotated_vertices)

        return Polygon(self, *self.rotations[angle])

class Polygon:
    def __init__(self, shape, centroid, vertices):
        self.shape = shape
        self.centroid = centroid
        self.vertices = vertices

    def translate(self, x, y):
        self.vertices += [x, y]
        self.centroid += [x, y]

    def translate_to(self, new_centroid):
        self.vertices += new_centroid - self.centroid
        self.centroid = new_centroid

    def resolve_overlap(self, other):
        trans = 0
        for p0, p1 in zip(self.vertices, np.roll(self.vertices, -1, axis=0)):
            for q0, q1 in zip(other.vertices,
                np.roll(other.vertices, -1, axis=0)):
                intersection = line_line_intersection(p0, p1, q0, q1)
                if intersection is not Intersection.Skew:
                    trans = max(trans, pir(p0, p1, q0, q1, intersection))

        return trans + TOL

    def resolve_nesting(self, other):
        vertex = self.vertices[0]
        intersections = 0
        trans = np.inf
        ray = np.array([0, 1])
        for q0, q1 in zip(other.vertices, np.roll(other.vertices, -1, axis=0)):
            point = ray_line_intersection_point(vertex, ray, q0, q1)
            if point is not Intersection.Skew:
                if np.isclose(point, q0).all():
                    if q1[1] < point[1]:
                        translation = min(trans, point[1] - vertex[1])
                        intersections += 1
                elif np.isclose(point, q1).all():
                    if q0[1] < point[1]:
                        translation = min(trans, point[1] - vertex[1])
                        intersections += 1
                else:
                    translation = min(trans, point[1] - vertex[1])
                    intersections += 1

        if intersections % 2 == 0:
            return 0
        else:
            return translation + TOL
