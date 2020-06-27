import json
import matplotlib.pyplot as plt
import os
from image_viewer import ImageViewer
import cv2


if __name__ == "__main__":
    show_filenames = [
        # 'evaluation_06_08_14_23_45.json',
        'evaluation_06_07_11_27.json',
        'evaluation_06_08_14_16_38.json',
        'evaluation_06_08_14_26_25.json',
    ]

    show_filenames = [f for f in os.listdir('evaluations') if f.endswith('.json')][-2:]

    all_dices = []
    all_filenames = None

    for i, FILENAME in enumerate(show_filenames):
        evaluation = {}
        with open(f'evaluations/{FILENAME}', 'r') as f:
            evaluation = json.load(f)
        dice_vals = evaluation['dice_vals']
        filenames = evaluation['filenames']
        mean = evaluation['mean']
        time = evaluation['time']
        kwargs = evaluation['kwargs']
        test_perc = evaluation['test_perc']

        all_dices.append(dice_vals)
        all_filenames = filenames if all_filenames is None else all_filenames

        pairs = list(zip(dice_vals, filenames))
        pairs.sort()
        dice_vals = [x for x, _ in pairs]
        filenames = [x for _, x in pairs]

        plt.plot(dice_vals, label=f'dice {i}')
        plt.axhline(y=mean, color='rgb'[i % 3], ls='--', label=f'mean {i}')
        print(f'{FILENAME:<32} mean={mean:.04f} time={time:.02f}s fps={len(filenames)/time:.02f} test_perc={test_perc} kwargs={kwargs}')
    plt.title('\n'.join(show_filenames))
    plt.legend()
    plt.show()

    if len(all_dices) == 2:
        worst = [(max(a-b, b-a), f) for a, b, f in zip(*all_dices, filenames)]
        worst.sort()
        iv = ImageViewer()
        for s, f in worst[-12:]:
            iv.add(cv2.imread(f'data_test/paintings_gt/imgs/{f}'), cmap='bgr', title=f'{s}')
        iv.show()