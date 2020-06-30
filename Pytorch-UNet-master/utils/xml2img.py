import cv2
import numpy as np
import xmltodict


def xml2img(filename):
    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())

    def get_points(obj):
        pts = []
        if 'polygon' in obj:
            polygon = obj['polygon']
            for pt in polygon['pt']:
                x = pt['x']
                y = pt['y']
                pts.append((x, y))
        return pts

    if 'annotation' in doc:
        height = int(doc['annotation']['imagesize']['nrows'])
        width = int(doc['annotation']['imagesize']['ncols'])

        mask = np.zeros((height, width), dtype=np.uint8)
        if 'object' in doc['annotation']:
            contours = []
            if isinstance(doc['annotation']['object'], list):
                for obj in doc['annotation']['object']:
                    contours.append(get_points(obj))
            else:
                contours.append(get_points(doc['annotation']['object']))

            for contour in contours:
                pts = np.array(contour).astype(np.int32)
                mask = cv2.fillPoly(mask, [pts], 255)
        return mask
    return None
