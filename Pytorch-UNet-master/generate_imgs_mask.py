import os
import numpy as np
import cv2 
from shutil import copyfile
import xmltodict

root_path = 'Pytorch-UNet-master/labelme'
dst_imgs_path = 'Pytorch-UNet-master/data/imgs' 
dst_masks_path = 'Pytorch-UNet-master/data/masks' 
# all_collections = [x[0] for x in os.walk(path)]
for r, d, f in os.walk(root_path):
    all_collections = d
    break

for collection in all_collections:
    collection_path = os.path.join(root_path, collection)
    all_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(collection_path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
    all_xml = [os.path.join(dp, f) for dp, dn, filenames in os.walk(collection_path) for f in filenames if os.path.splitext(f)[1] == '.xml']
    for img_path, xml_path in zip(all_images, all_xml):
        img_filename = os.path.basename(img_path)
        copyfile(img_path, os.path.join(dst_imgs_path, img_filename))
        with open(xml_path) as fd:
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

        if 'annotation' in doc and 'object' in doc['annotation']:
            contours = []
            if isinstance(doc['annotation']['object'], list):
                for obj in doc['annotation']['object']:
                    contours.append(get_points(obj))
            else:
                contours.append(get_points(doc['annotation']['object']))
        
        height = int(doc['annotation']['imagesize']['nrows'])
        width =  int(doc['annotation']['imagesize']['ncols'])

        mask = np.zeros((height, width), dtype=np.uint8)
        for contour in contours:
            pts = np.array(contour).astype(np.int32)
            mask = cv2.fillPoly(mask, [pts], 255)
        # cv2.imshow('', mask)
        # cv2.waitKey(1)
        cv2.imwrite(os.path.join(dst_masks_path, img_filename).replace('.jpg', '.png'), mask) 
        # print(len(contours))