import cv2
import numpy as np
import math

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

def groups_by_angle(lines, k=2):
    """
    Divide in cluster the lines based on its angle
    
    Parameters
    ----------
    lines : list
        list of all lines found in the image
    k : int
        number of clusters

    Returns
    -------
    list
        returns a list of k list, where k is the number of clusters. Each line is in the list
        of its cluster
    """
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (default_criteria_type, 10, 1.0)

    angles = np.array([line[0][1] for line in lines])
    """
    Why 2*angle? (angle is in range [0, pi])

    Normally:   
        if line is vertical (pi/2)          sin=0   cos=1
        if line is oblique (pi/4)           sin=0.5 cos=0.5
        if line is oblique (3/4 pi)         sin=0.5 cos=-0.5
        if line is horizontal (0 or pi)     sin=1   cos=0

    Multiplied by 2:
        if line is vertical (pi/2)          sin=0 cos=-1
        if line is oblique (pi/4)           sin=1 cos=0
        if line is oblique (3/4 pi)         sin=-1 cos=0
        if line is horizontal (0 or pi)     sin=0 cos=1   
    
    Actually I think that the 2* multiplication is not so crucial,
    but in the script that I used as "inspiration" use it,
    so I did the same. 
    """
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)

    groups = [ [] for _ in range(k) ] 
    for label, line in zip(labels, lines):
        groups[label].append(line)

    return groups

def find_all_intersections(groups):
    """
    Find all possible points of interection between all lines of all groups
    
    Parameters
    ----------
    groups : list
        list of groups of lines

    Returns
    -------
    list
        returns a list of points [x, y]
    """
    points = []
    for index, group in enumerate(groups[:-1]):
        for line in group:
            for group2 in groups[index + 1:]:
                for line2 in group2:
                    points.append(intersection(line, line2))
    return points

def find_four_corners(points):
    """
    Find 4 possible points corners using kmeans
    
    Parameters
    ----------
    points : list
        list of points [x, y]

    Returns
    -------
    list
        returns a list of the corners points [x, y]
    """
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (default_criteria_type, 10, 1.0)
    pts = np.array([[point[0], point[1]] for point in points], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]

    corners = []
    for center in centers:
        x = int(np.round(center[0]))
        y = int(np.round(center[1]))
        corners.append([x, y])
    return corners

def draw_corners(img, corners):
    kernel = np.ones((5, 5), np.uint8)
    debug_img = cv2.dilate(img.copy(), kernel)
    canvas = np.stack((debug_img,)*3, axis=-1)
    color = np.random.randint(256, size=3).tolist()
    for point in corners:
        cv2.circle(canvas, (point[0], point[1]), 20, color, -1)
    return canvas

def find_corners(input, debug=False):
    """
    Given a list of lines it finds the 4 corners of the painting
    
    Parameters
    ----------
    lines : list
        list of lines

    Returns
    -------
    list
        returns a list of the corners points [x, y]
    """
    img, lines = input
    if lines is None:
        return None
    groups = groups_by_angle(lines)
    points = find_all_intersections(groups)
    corners = find_four_corners(points)
    if debug:
        canvas = draw_corners(img, corners)
        return corners, canvas
    else:
        return corners


if __name__ == '__main__':
    from pipeline import Pipeline
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(10)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()