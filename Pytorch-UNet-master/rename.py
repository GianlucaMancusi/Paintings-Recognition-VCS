import os
import sys

# [os.rename(a, a.upper()) for a in sys.argv[1:]]
for a in sys.argv[1:]:
    base = os.path.basename(a)
    # print(a.replace(base, base.upper()))
    if '.xml' in a.lower():
        os.rename(a, a.replace('.XML', '.xml'))
    elif '.jpg' in a.lower():
        os.rename(a, a.replace('.JPG', '.jpg'))

