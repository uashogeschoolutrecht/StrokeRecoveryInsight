

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2)
    x = (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2)
    y = (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2)
    z = (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(v1, q1):
    v0xyz = q_mult(q_mult(q1,v1), q_conjugate(q1))

    return [v0xyz[1],v0xyz[2],v0xyz[3]]