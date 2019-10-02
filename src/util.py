from enum import Enum
from copy import deepcopy

import numpy as np
from scipy.spatial import ConvexHull

TOL = 1e-5

class Intersection(Enum):
    Collinear = 0
    Oblique = 1
    Skew = 2

def find_centroid(vertices):
    return np.mean(vertices, axis=0)

def find_bbox(vertices):
    return [np.amin(vertices, axis=0), np.amax(vertices, axis=0)]

def interval_intersection(a0, a1, b0, b1):
    return min(a1, b1) > max(a0, b0)

def line_line_intersection(p0, p1, q0, q1):
    if not interval_intersection(min(p0[0], p1[0]) - TOL,
        max(p0[0], p1[0]) + TOL, min(q0[0], q1[0]) - TOL,
        max(q0[0], q1[0]) + TOL) or \
        not interval_intersection(min(p0[1], p1[1]) - TOL,
            max(p0[1], p1[1]) + TOL,
            min(q0[1], q1[1]) - TOL,
            max(q0[1], q1[1]) + TOL):
        return Intersection.Skew

    r = p1 - p0
    s = q1 - q0
    c = np.cross(r, s)
    if np.isclose(c, 0).all():
        if not np.isclose(np.cross((q0 - p0), r), 0).all():
            return Intersection.Skew

        length = np.dot(r, r)
        t0 = np.dot((q0 - p0), r) / length
        t1 = t0 + np.dot(s, r) / length
        if min(max(t0, t1), 1) > max(min(t0, t1), 0):
            return Intersection.Collinear
    else:
        t = np.cross((q0 - p0), s) / c
        u = np.cross((q0 - p0), r) / c
        if 0 <= t <= 1 and 0 <= u <= 1:
            return Intersection.Oblique

    return Intersection.Skew

def ray_line_intersection_point(p0, r, q0, q1):
    s = q1 - q0
    c = np.cross(r, s)
    if not np.isclose(c, 0).all():
        t = np.cross((q0 - p0), s) / c
        u = np.cross((q0 - p0), r) / c
        if 0 <= t and 0 <= u <= 1:
            return p0 + r * t

    return Intersection.Skew

def pir(p0, p1, q0, q1, intersection):
    t = time()
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

        return Polygon(self, *deepcopy(self.rotations[angle]))

class Polygon:
    def __init__(self, shape, centroid, vertices):
        self.shape = shape
        self.centroid = centroid
        self.vertices = vertices
        self.bbox = find_bbox(self.vertices)
        self.translation = np.array([0, 0], dtype=float)

    def translate(self, x, y):
        self.vertices += [x, y]
        self.centroid += [x, y]
        self.bbox[0] += [x, y]
        self.bbox[1] += [x, y]
        self.translation += [x, y]

    def resolve_overlap(self, other):
        min0, max0 = self.bbox
        min1, max1 = other.bbox
        if not interval_intersection(min0[0], max0[0], min1[0], max1[0]) or \
            not interval_intersection(min0[1], max0[1], min1[1], max1[1]):
            return 0

        trans = 0
        for p0, p1 in zip(self.vertices, np.roll(self.vertices, -1, axis=0)):
            for q0, q1 in zip(other.vertices,
                np.roll(other.vertices, -1, axis=0)):
                intersection = line_line_intersection(p0, p1, q0, q1)
                if intersection is not Intersection.Skew:
                    trans = max(trans, pir(p0, p1, q0, q1, intersection))
                    if trans > 0:
                        return trans + TOL

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

class Sheet:
    def __init__(self, height, increment):
        self.height = height
        self.increment = increment
        self.locked_shapes = []
        self.x = 0

    def resolve_all(self, polygon):
        for other in self.locked_shapes:
            trans = polygon.resolve_overlap(other)
            if trans <= TOL:
                trans = polygon.resolve_nesting(other)
            if trans > TOL:
                return trans
        return 0

    def place_first(self, shape, rotations):
        min_area = np.inf
        orientation = None
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / rotations):
            polygon = shape.get_rotation(angle)
            (x0, y0), (x1, y1) = find_bbox(polygon.vertices)
            area = (x1 - x0) * (y1 - y0)
            if area < min_area:
                min_area = area
                orientation = polygon
        self.locked_shapes.append(orientation)

    def cost_to_place(self, polygon):
        full = np.unique(np.array([vertex for p in self.locked_shapes \
            for vertex in p.vertices]), axis=0)
        area = ConvexHull(full).area
        return ConvexHull(np.unique(np.concatenate((full,
            polygon.vertices)), axis=0)).area - area

    def bottom_left_place(self, shape, rotations):
        min_cost = np.inf
        orientation = None
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / rotations):
            inc = 0
            polygon = shape.get_rotation(angle)
            for other in reversed(self.locked_shapes):
                if other.shape is polygon.shape:
                    polygon.translate(*other.translation)
                    break
            else:
                polygon.translate(self.x, 0)
            while True:
                translation = self.resolve_all(polygon)
                while translation > TOL:
                    polygon.translate(0, translation)
                    translation = self.resolve_all(polygon)
                    y_max = polygon.bbox[1][1]
                    if y_max > self.height:
                        mins = polygon.bbox[0]
                        polygon.translate(self.increment, -mins[1])
                        break
                else:
                    break

            cost = self.cost_to_place(polygon)
            if cost < min_cost:
                min_cost = cost
                orientation = polygon

        self.x = orientation.translation[0]
        self.locked_shapes.append(orientation)

    def bottom_left_fill(self, shapes, rotations):
        shape = shapes[0]
        self.place_first(shape, rotations)
        for shape in shapes[1:]:
            self.bottom_left_place(shape, rotations)
