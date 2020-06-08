import json
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    show_filenames = [
        # 'evaluation_06_08_14_23_45.json',
        'evaluation_06_07_11_27.json',
        'evaluation_06_08_14_16_38.json',
        'evaluation_06_08_14_26_25.json',
    ]

    show_filenames = [f for f in os.listdir('evaluations') if f.endswith('.json')][-12:]

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