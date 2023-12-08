import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
plt.rcParams.update({
    "pdf.fonttype": 42,
})
# figure(figsize=(3.5, 2.5))   # max width is 3.5 for single column
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)

DATA = {
    'Giraffe': {
        'height': [
            5.486,  # 000_004.pcd
            5.552,  # 001_005.pcd
            5.486,  # 002_006.pcd
            5.490,  # 003_007.pcd
            5.496,  # 004_008.pcd
            5.514,  # 005_009.pcd
            5.490,  # 006_010.pcd
            5.490,  # 007_011.pcd
            5.490,  # 008_012.pcd
            5.490,  # 009_013.pcd
            5.461,  # 010_014.pcd
            5.466,  # 011_015.pcd
            5.497,  # 012_016.pcd
            5.497,  # 013_017.pcd
            5.503,  # 014_018.pcd
            5.503,  # 015_019.pcd
            5.525,  # 016_020.pcd
        ],
        'nose–r_eye': [0.4652536603408275, 0.39569287382628676, 0.3755183911773079, 0.41043636537821554, 0.4144803502397298, 0.4519357405261681, 0.4309165235544584, 0.38877652844654165, 0.4319884260021488, 0.43260797730882744, 0.4376732754198414, 0.416647312862881, 0.416647312862881, 0.4174170916380571, 0.44269428862236343, 0.40908924521985734, 0.4045814838811479],
        'r_eye–neck': [2.221215764337507, 2.1960217089970144, 2.20547772497857, 2.2058424003721027, 2.2684799288898247, 2.1778501275374276, 2.263728702693725, 2.254683339729913, 2.2183635877997685, 2.252977308871741, 2.213031529525996, 2.228959773850837, 2.2485316263726918, 2.225180447500645, 2.228959773850837, 2.2157820375040975, 2.226464766468422],
        'neck–hip': [1.314890227608899, 1.2739539739002783, 1.2818091423828413, 1.3734725539087647, 1.2318217897860926, 1.3756441537140625, 1.2524340072025428, 1.2851091238643584, 1.2952360015840672, 1.2830576893089696, 1.3455456288632175, 1.2949667188304979, 1.2876058970935849, 1.252011517963586, 1.2949667188304979, 1.3148126375784417, 1.3618488894443541],
    },
    'Martial eagle': {
        'height': [
            0.494,  # 000_004.pcd
            0.501,  # 001_005.pcd
            0.493,  # 002_006.pcd
            0.493,  # 003_007.pcd
            0.493,  # 004_008.pcd
            0.488,  # 005_009.pcd
            0.489,  # 006_010.pcd
            0.492,  # 007_011.pcd
            0.497,  # 008_012.pcd
            0.497,  # 009_013.pcd
            0.497,  # 010_014.pcd
            0.497,  # 011_015.pcd
            0.497,  # 012_016.pcd
            0.493,  # 013_017.pcd
            0.493,  # 014_018.pcd
            0.493,  # 015_019.pcd
            0.490,  # 016_020.pcd
        ],
        'nose–r_eye': [0],
        'r_eye–neck': [0],
        'neck–hip': [0],
    },
}


def main():
    length_kinds = list(DATA[list(DATA.keys())[0]].keys())
    x = np.arange(len(length_kinds))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(
        figsize=(3.5, 2.5),
        layout='constrained'
    )
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    for animal, data in DATA.items():
        # collect values
        ys = [np.mean(data[l]) for l in length_kinds]
        y_errs = [np.std(data[l]) for l in length_kinds]
        # draw bars
        offset = width * multiplier
        rects = ax.bar(
            x + offset, ys,
            yerr=y_errs,
            ecolor='black',
            capsize=10,
            width=width,
            label=animal)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Length (m)')
    ax.set_xticks(x + width/2, length_kinds)
    ax.legend(loc='upper right')
    # ax.set_ylim(0, 250)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for fmt in ['svg', 'pdf']:
        plt.savefig(f"results/output.{fmt}", format=fmt, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
