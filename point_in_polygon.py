

def work(args):
    pt, poly_group = args
    out = []
    for i, polys in enumerate(poly_group):
        in_feature = (pt.within(poly) for poly in polys)
        if any(in_feature):
            out.append(i)
    return out
