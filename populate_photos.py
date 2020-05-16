import os
import cv2
import time
videos_path = 'dataset/videos'
photos_path = 'dataset/photos'
occurrence = 45

t0 = time.time()

"""
This script read all videos in the directory "dataset/videos" and save
one frame every 'occurrence' (es. 15) in the directory
"dataset/photos/VIDEOS_DIR/VIDEO_NAME"
"""

if not os.path.exists(photos_path):
    os.makedirs(photos_path)
    print('Created photos directory')
else:
    print('Photos directory already exists')

print('Saving 1 frame every {}'.format(occurrence))

# Separator
print()
print('-' * 50)
print()

for subdir, dirs, files in os.walk(videos_path):
    videos_dir_name = subdir.replace('/', '\\').split('\\')[-1]

    if videos_dir_name == 'videos':
        continue

    print('Opened directory "{}"'.format(videos_dir_name))

    dst_dir = os.path.join(photos_path, videos_dir_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print('Created the directory "{}"'.format(dst_dir))
    else:
        print('The directory "{}" already exists'.format(dst_dir))
    
    for video in files:
        video_name = os.path.splitext(video)[0]
        video_dir = os.path.join(dst_dir, video_name)

        print('\tOpened video "{}"'.format(video))

        # if not os.path.exists(video_dir):
        #     os.makedirs(video_dir)
        #     print('\tCreated the directory "{}"'.format(video_dir))
        # else:
        #     print('\tThe directory "{}" already exists'.format(video_dir))
        
        video_path = os.path.join(subdir, video)
        videoCapture = cv2.VideoCapture(video_path)
        success, image = videoCapture.read()
        count = 0
        saved = 0 
        print('\treading:\t', end='')
        frame_count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_number = 0
        while success and frame_number <= frame_count:
            cv2.imwrite(os.path.join(dst_dir, '{}_{:03d}.jpg'.format(video_name, saved)), image)
            saved += 1
            print('x', end='')

            # do stuff

            frame_number += occurrence
            videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = videoCapture.read()
            if saved % 90 == 0:
                print('  {}\n\t\t\t'.format(count), end='')
        print('\tSaved frames: {}'.format(saved))
        print('\tElapsed time: {:.2f} s\n\n'.format(time.time() - t0))
