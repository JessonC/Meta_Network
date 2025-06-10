"""Generate grid-based resistor network for a PCB.

This module builds a resistor network from a polygonal board outline and
optional via polygons. The board is discretized on a square grid with
spacing `d`. Each grid cell is clipped against the actual PCB outline so
that partial coverage near edges and vias is taken into account. Nodes are
created for cells with non-zero overlap area and resistive connections are
weighted by the average overlap area of adjacent cells.

Usage:
    network, points = build_resistor_network(board, vias, d, resistivity)

`network` is a dict mapping node index to a dict of neighbour indices and
resistances. `points` holds the coordinates of each node by index.
"""

from typing import List, Tuple, Dict
import math

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Point = Tuple[float, float]
Polygon = List[Point]


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """Return True if point is inside polygon using ray casting."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def bounding_box(poly: Polygon) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(poly: Polygon) -> float:
    """Return the signed area of a polygon."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    j = -1
    for i, (x, y) in enumerate(poly):
        xj, yj = poly[j]
        area += xj * y - x * yj
        j = i
    return 0.5 * area


def circle_polygon(cx: float, cy: float, r: float, segments: int = 20) -> Polygon:
    """Return a polygon approximating a circle."""
    pts: Polygon = []
    for i in range(segments):
        theta = 2 * math.pi * i / segments
        pts.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pts


def clip_polygon_with_rect(
    poly: Polygon, xmin: float, ymin: float, xmax: float, ymax: float
) -> Polygon:
    """Clip polygon with axis aligned rectangle using Sutherland-Hodgman."""

    def inside_left(pt: Point) -> bool:
        return pt[0] >= xmin

    def inside_right(pt: Point) -> bool:
        return pt[0] <= xmax

    def inside_bottom(pt: Point) -> bool:
        return pt[1] >= ymin

    def inside_top(pt: Point) -> bool:
        return pt[1] <= ymax

    def intersect(p1: Point, p2: Point, x=None, y=None) -> Point:
        x1, y1 = p1
        x2, y2 = p2
        if x is not None:
            if x2 != x1:
                t = (x - x1) / (x2 - x1)
                return (x, y1 + t * (y2 - y1))
            return (x, y1)
        if y is not None:
            if y2 != y1:
                t = (y - y1) / (y2 - y1)
                return (x1 + t * (x2 - x1), y)
            return (x1, y)
        raise ValueError("Either x or y must be provided")

    def clip(poly: Polygon, inside, edge_value, vertical=False) -> Polygon:
        if not poly:
            return poly
        out: Polygon = []
        prev = poly[-1]
        prev_in = inside(prev)
        for curr in poly:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    if vertical:
                        out.append(intersect(prev, curr, x=edge_value))
                    else:
                        out.append(intersect(prev, curr, y=edge_value))
                out.append(curr)
            elif prev_in:
                if vertical:
                    out.append(intersect(prev, curr, x=edge_value))
                else:
                    out.append(intersect(prev, curr, y=edge_value))
            prev = curr
            prev_in = curr_in
        return out

    poly = clip(poly, inside_left, xmin, vertical=True)
    poly = clip(poly, inside_right, xmax, vertical=True)
    poly = clip(poly, inside_bottom, ymin, vertical=False)
    poly = clip(poly, inside_top, ymax, vertical=False)
    return poly


def build_resistor_network(
    board: Polygon,
    vias: List[Polygon],
    d: float,
    resistivity: float,
) -> Tuple[Dict[int, Dict[int, float]], List[Point]]:
    """Create resistor network for a board.

    Parameters
    ----------
    board: Polygon defining the outer board boundary.
    vias: List of polygons representing voids in the board.
    d: Grid spacing.
    resistivity: Sheet resistivity of the board material.
    """
    min_x, min_y, max_x, max_y = bounding_box(board)

    # Expand bounds slightly to cover edge cells
    min_x -= d
    min_y -= d
    max_x += d
    max_y += d

    grid_index: Dict[Tuple[int, int], int] = {}
    points: List[Point] = []
    areas: List[float] = []
    idx = 0

    j = 0
    y = min_y + d / 2
    while y <= max_y:
        i = 0
        x = min_x + d / 2
        while x <= max_x:
            xmin, ymin = x - d / 2, y - d / 2
            xmax, ymax = x + d / 2, y + d / 2
            cell_poly = clip_polygon_with_rect(board, xmin, ymin, xmax, ymax)
            area = abs(polygon_area(cell_poly))
            for via in vias:
                via_clip = clip_polygon_with_rect(via, xmin, ymin, xmax, ymax)
                area -= abs(polygon_area(via_clip))
            if area > 0:
                grid_index[(i, j)] = idx
                points.append((x, y))
                areas.append(area)
                idx += 1
            i += 1
            x += d
        j += 1
        y += d

    network: Dict[int, Dict[int, float]] = {n: {} for n in range(idx)}
    for (i, j), n1 in grid_index.items():
        neighbours = [
            (i + 1, j),  # right
            (i - 1, j),  # left
            (i, j + 1),  # up
            (i, j - 1),  # down
        ]
        for nb in neighbours:
            n2 = grid_index.get(nb)
            if n2 is None:
                continue
            cross_area = (areas[n1] + areas[n2]) / 2.0
            if cross_area <= 0:
                continue
            R = resistivity * d / cross_area
            network[n1][n2] = R
    return network, points


def demo():
    """Demonstrate network generation on an irregular board with vias.

    The demo builds a larger trapezoidal board with ten circular vias. The
    resistor network is written to ``demo_network.svg`` where edge colours
    correspond to resistance values.
    """

    # Larger irregular trapezoid board
    board = [
        (0.0, 0.0),
        (16.0, 0.0),
        (20.0, 12.0),
        (12.0, 20.0),
        (0.0, 16.0),
    ]

    # Ten small circular vias
    via_centers = [
        (4.0, 4.0),
        (7.0, 5.0),
        (10.0, 10.0),
        (14.0, 14.0),
        (8.0, 6.0),
        (12.0, 4.0),
        (16.0, 8.0),
        (10.0, 14.0),
        (2.0, 12.0),
        (6.0, 16.0),
    ]
    # Slightly smaller vias for clarity
    r = 0.4
    vias = [circle_polygon(cx, cy, r, 20) for cx, cy in via_centers]

    network, pts = build_resistor_network(board, vias, 1.0, 0.02)
    print("Number of nodes:", len(pts))

    # Show a connection near the centre of the board
    for idx, pt in enumerate(pts):
        if 9.0 < pt[0] < 11.0 and 9.0 < pt[1] < 11.0:
            print("Node", idx, "at", pt)
            print("Neighbours:", network[idx])
            break

    # Collect edges for visualisation
    edges = []
    resistances = []
    for n1, conn in network.items():
        for n2, R in conn.items():
            if n1 < n2:
                edges.append((pts[n1], pts[n2]))
                resistances.append(R)

    if not edges:
        print("No edges to display")
        return

    min_r = min(resistances)
    max_r = max(resistances)

    def colour_for(r: float) -> str:
        """Return an RGB string from blue (low) to red (high)."""
        t = (r - min_r) / (max_r - min_r + 1e-12)
        red = int(255 * t)
        blue = int(255 * (1 - t))
        return f"rgb({red},0,{blue})"

    min_x, min_y, max_x, max_y = bounding_box(board)
    width = max_x - min_x
    height = max_y - min_y

    fname = "demo_network.svg"
    with open(fname, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x-1} {min_y-1} {width+2} {height+2}" '
            f'width="{width+200}" height="{height+200}">\n'
        )
        style = (
            "<style>"
            ".board{fill:none;stroke:black;}"
            ".via{fill:none;stroke:red;stroke-dasharray:2;}"
            ".node{fill:black;stroke:none;}"
            ".edge{stroke-width:0.1;}"
            "</style>\n"
        )
        f.write(style)
        pts_str = " ".join(f"{x},{y}" for x, y in board)
        f.write(f'<polygon class="board" points="{pts_str}"/>\n')
        for cx, cy in via_centers:
            f.write(f'<circle class="via" cx="{cx}" cy="{cy}" r="{r}"/>\n')
        for (p1, p2), r in zip(edges, resistances):
            color = colour_for(r)
            f.write(
                f'<line class="edge" x1="{p1[0]}" y1="{p1[1]}" '
                f'x2="{p2[0]}" y2="{p2[1]}" stroke="{color}"/>\n'
            )
        for x, y in pts:
            f.write(f'<circle class="node" cx="{x}" cy="{y}" r="0.1"/>\n')
        f.write("</svg>\n")

    print("Saved", fname)


if __name__ == "__main__":
    demo()
