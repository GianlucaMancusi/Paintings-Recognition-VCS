import os
import sys
from data_test.standard_samples import all_files_in

# [os.rename(a, a.upper()) for a in sys.argv[1:]]
masks_dir = 'data_test/paintings_gt/masks'

for a in all_files_in(masks_dir):
    base = os.path.basename(a)
    base = base.split('.')[0]
    # print(base)
    print(a.replace(base, base.upper()))
    os.rename(a, a.replace(base, base.upper()))
    # if '.xml' in a.lower():
    #     os.rename(a, a.replace('.XML', '.xml'))
    # elif '.jpg' in a.lower():
    #     os.rename(a, a.replace('.JPG', '.jpg'))

