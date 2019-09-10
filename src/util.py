from typings import List

class Vertex:
    def __init__(self, x: float, y: float):
        pass

class Polygon:
    def __init__(self: Polygon, vertices: List[Vertex]):
        pass

    def translate(self, x, y) -> Polygon:
        pass

    def detect_overlap(self: Polygon, other: Polygon) -> List[tuple]:
        pass

    def is_nested(self: Polygon, other: Polygon) -> bool:
        pass

    def resolve_overlap(self: Polygon, intersections: List[tuple]) -> Polygon:
        pass

    def resolve_nesting(self: Polygon, other: Polygon) -> Polygon:
        pass

def rotations(vertices: List[Vertex], rotations: int) -> List[Polygons]:
    pass