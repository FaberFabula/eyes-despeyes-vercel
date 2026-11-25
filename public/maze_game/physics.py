"""Bodies, collision filters, and spatial queries."""

from math import inf, radians

import pymunk
from pymunk import Body, Poly, Segment, Shape, ShapeFilter, Space, Vec2d
from pymunk import PointQueryInfo as PointHit, SegmentQueryInfo as SweepHit

SOLID = 0b0001
ACTOR = 0b0010
GEAR_CATEGORY = 0b0100
SPIKE_CATEGORY = 0b1000
GEAR_TYPE = 1
ACTOR_TYPE = 2
SPIKE_TYPE = 3
ALL_CATEGORIES = 0xFFFFFFFF


class Physics:
    def __init__(self, damping: float = 0.45):
        self.space = pymunk.Space()
        self.space.damping = damping

    def add_bounds(self, width: float, height: float) -> None:
        corners = ((0, 0), (width, 0), (width, height), (0, height))
        for start, end in zip(corners, (*corners[1:], corners[0])):
            edge = Segment(self.space.static_body, start, end, 1)
            edge.filter = ShapeFilter(categories=SOLID)
            self.space.add(edge)

    def add_wall(self, corners: list[tuple[float, float]]) -> Poly:
        shape = Poly(self.space.static_body, corners)
        shape.filter = ShapeFilter(categories=SOLID)
        self.space.add(shape)
        return shape

    def add_actor(self, position: Vec2d, radius: float, apply_velocity) -> tuple[Body, Shape]:
        body = Body((radius / 10) ** 2, inf)
        body.position = position
        body.velocity_func = apply_velocity
        shape = pymunk.Circle(body, radius)
        shape.filter = ShapeFilter(categories=ACTOR)
        shape.collision_type = ACTOR_TYPE
        self.space.add(body, shape)
        return body, shape

    def add_fixed_target(self, position: Vec2d, radius: float) -> Shape:
        shape = pymunk.Circle(self.space.static_body, radius, position)
        shape.filter = ShapeFilter(categories=ACTOR)
        self.add_shape(shape)
        return shape

    def add_shape(self, shape: Shape) -> None:
        self.space.add(shape)

    def remove_shape(self, shape: Shape) -> None:
        self.space.remove(shape)

    def remove_actor(self, shape: Shape) -> None:
        self.space.remove(shape, shape.body)

    def contains(self, shape: Shape) -> bool:
        return shape.space is self.space

    def refresh_shape(self, shape: Shape) -> None:
        self.space.reindex_shape(shape)

    def refresh_body(self, body: Body) -> None:
        self.space.reindex_shapes_for_body(body)

    def resize_actor(self, shape: pymunk.Circle, radius: float, *, mass_radius: float | None = None) -> None:
        if mass_radius is not None:
            shape.body.mass = (mass_radius / 10) ** 2
        shape.unsafe_set_radius(radius)
        self.refresh_shape(shape)

    def first_hit(self, start: Vec2d, end: Vec2d, radius: float, mask: int) -> SweepHit | None:
        return self.space.segment_query_first(start, end, radius, ShapeFilter(mask=mask))

    def sweep(self, start: Vec2d, end: Vec2d, radius: float, mask: int) -> list[SweepHit]:
        return self.space.segment_query(start, end, radius, ShapeFilter(mask=mask))

    def near_solid(self, point: Vec2d, distance: float) -> bool:
        return self.space.point_query_nearest(point, distance, ShapeFilter(mask=SOLID)) is not None

    def solid_clearance(self, point: Vec2d, limit: float) -> float:
        hit = self.space.point_query_nearest(point, limit, ShapeFilter(mask=SOLID))
        return hit.distance if hit is not None else limit

    def shapes_near(self, origin: Vec2d, radius: float) -> list[Shape]:
        box = pymunk.BB(origin.x - radius, origin.y - radius, origin.x + radius, origin.y + radius)
        return self.space.bb_query(box, ShapeFilter())

    def suspend_collisions(self, shape: Shape, *, categories: int = ALL_CATEGORIES) -> ShapeFilter:
        previous = shape.filter
        shape.filter = ShapeFilter(categories=categories, mask=0)
        return previous

    def restore_collisions(self, shape: Shape, original: ShapeFilter) -> None:
        shape.filter = original
        self.refresh_shape(shape)

    def placement_probe(self, radius: float, size: tuple[float, float] | None = None, *, angle: float = 0):
        body = Body(body_type=Body.KINEMATIC)
        body.angle = angle
        shape = pymunk.Circle(body, radius) if size is None else Poly.create_box(body, size)
        shape.filter = ShapeFilter(mask=SOLID | ACTOR | GEAR_CATEGORY)

        def clear_at(position: Vec2d) -> bool:
            body.position = position
            shape.cache_bb()
            return not any(not hit.shape.sensor for hit in self.space.shape_query(shape))

        return clear_at

    def step(self, dt: float) -> None:
        self.space.step(dt)


def add_item_body(space: pymunk.Space, spec, position: Vec2d, angle: float
                  ) -> tuple[Body, Poly, tuple[Segment, ...]]:
    size = spec.length, spec.width
    body = Body(spec.mass, pymunk.moment_for_box(spec.mass, size))
    body.position = position
    body.angle = angle
    jaws = ()
    if spec.effect == "catcher":
        left, right = -spec.length / 2, spec.length * 0.25
        shape = Poly(body, ((left, -3), (right, -3), (right, 3), (left, 3)), radius=1.5)
        jaws = tuple(Segment(body, *catcher_jaw_line(spec, side, 0), 2.25) for side in (-1, 1))
    else:
        shape = Poly.create_box(body, size, radius=1.5)
    for collider in (shape, *jaws):
        collider.filter = ShapeFilter(categories=GEAR_CATEGORY)
        collider.collision_type = GEAR_TYPE
        collider.elasticity = 0.2
        collider.friction = 0.5
    space.add(body, shape, *jaws)
    return body, shape, jaws


def catcher_jaw_line(spec, side: int, closure: float) -> tuple[Vec2d, Vec2d]:
    hinge = Vec2d(spec.length * 0.25, 0)
    tip = Vec2d(spec.length * 0.25, side * (spec.width / 2 - 2.25))
    return hinge, hinge + tip.rotated(-side * radians(25) * closure)
