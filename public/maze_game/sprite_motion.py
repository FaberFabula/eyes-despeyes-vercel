from math import cos, sin, tau


def mimic_legs(count: int, time: float, radius: float):
    """Generate irregular jointed legs in local sprite coordinates."""
    for index in range(count):
        angle = index * tau / count + 0.12 * sin(index * 2.2)
        stride = sin(time + index * 2.8)
        length = 1.93 + 0.28 * sin(index * 3.91) + 0.13 * stride
        knee = angle + 0.21 * cos(index * 1.4)
        foot = angle + 0.18 * stride
        yield (
            (cos(angle) * radius * 0.49, sin(angle) * radius * 0.42),
            (cos(knee) * radius * 1.16, sin(knee) * radius * 0.86 - radius * 0.25),
            (cos(foot) * radius * length, sin(foot) * radius * length * 0.73 + radius * 0.18),
        )
