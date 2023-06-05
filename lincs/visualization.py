# Copyright 2023 Vincent Jacques

import matplotlib.pyplot as plt


def visualize_model(problem, model, alternatives, alternatives_count, out):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")

    xs = [criterion.name for criterion in problem.criteria]
    ys = [list(boundary.profile) for boundary in model.boundaries]
    ys.append([1] * len(xs))
    unstacked_ys = [ys[0]]
    for ys1, ys2 in zip(ys[1:], ys[:-1]):
        unstacked_ys.append([y1 - y2 for y1, y2 in zip(ys1, ys2)])
    collections = ax.stackplot(
        xs, unstacked_ys,
        labels=[category.name for category in problem.categories],
        alpha=0.4,
    )
    colors = [collection.get_facecolor() for collection in collections]

    # @todo Rethink class Alternative to also expose its assigned category *index*, not just category name
    category_indexes = {category.name: index for index, category in enumerate(problem.categories)}

    if alternatives:
        for alternative in alternatives.alternatives[:alternatives_count]:
            color = colors[category_indexes[alternative.category]]
            ax.plot(
                xs, list(alternative.profile),
                "o--",
                label=alternative.name,
                color=color,
                alpha=1,
            )

    ax.legend()
    ax.set_ylim(0, 1)

    fig.savefig(out, format="png", dpi=100)
    plt.close(fig)
