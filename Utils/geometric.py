# ======================================================
# Geometric utils
# ======================================================
from scipy import spatial

# ======================================================
# findClosestPoints, taken from
#   https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
#   The approach is generally to first use the point data to build up a k-d tree. The computational complexity
# of that is on the order of N log N, where N is the number of data points. Range queries and nearest neighbour
# searches can then be done with log N complexity. This is much more efficient than simply cycling through all
# points (complexity N).
#   Thus, if you have repeated range or nearest neighbor queries, a k-d tree is highly recommended.
def findClosestPoints(reference, target, N=1):
    tree = spatial.cKDTree(reference)
    dist, indexes = tree.query(target, k=N)
    indexes = indexes.reshape((-1,N))
    dist = dist.reshape((-1,N))
    return indexes, dist