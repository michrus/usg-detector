from typing import List, Tuple


def coords_from_bound_rect(x: int, y: int, w: int, h: int) -> List[Tuple[int]]:
    x1 = x
    y1 = y
    x2 = x+w
    y2 = y+h
    return [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
