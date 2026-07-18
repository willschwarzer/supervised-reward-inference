from math import sqrt

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def norm(x):
    return sqrt(dot_product(x, x))

def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]

def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]

def project_onto_plane_unit_sphere(x, n):
    return normalize(project_onto_plane(x, n))

def hypercone_from_file(filename):
    with open(filename, "r") as f:
        firstline = True
        planes = []
        for line in f:
            if firstline:
                firstline = False
                continue
            planes.append(Plane([float(x.strip()) for x in line.strip().split(" ")]))
        return Hypercone(planes)

class Plane:
    def __init__(self, normal):
        self.normal = normal

    def __repr__(self):
        return f"Plane({', '.join([str(x) for x in self.normal])})"

    def __str__(self):
        return self.__repr__()

class Hypercone:
    def __init__(self, planes):
        self.planes = planes

    def __repr__(self):
        return f"Hypercone[{len(self.planes)}] {{{', '.join([repr(x) for x in self.planes])}}}"

    def __str__(self):
        return f"Hypercone[{len(self.planes)}]"
