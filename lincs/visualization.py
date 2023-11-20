# Copyright 2023 Vincent Jacques

import matplotlib.pyplot as plt


def visualize_model(problem, model, alternatives, alternatives_count, out):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")

    xs = [criterion.name for criterion in problem.criteria]
    boundary_profiles = [[] for _ in problem.ordered_categories[1:]]
    for criterion_index, criterion in enumerate(problem.criteria):
        assert criterion.is_real
        for boundary_index in range(len(problem.ordered_categories) - 1):
            acc_vals = model.accepted_values[criterion_index]
            boundary_profiles[boundary_index].append(acc_vals.real_thresholds[boundary_index])
    ys = [
        normalize_profile(problem.criteria, boundary_profile)
        for boundary_profile in boundary_profiles
    ]
    ys.append([1] * len(xs))
    unstacked_ys = [ys[0]]
    for ys1, ys2 in zip(ys[1:], ys[:-1]):
        unstacked_ys.append([y1 - y2 for y1, y2 in zip(ys1, ys2)])
    collections = ax.stackplot(
        xs, unstacked_ys,
        labels=[category.name for category in problem.ordered_categories],
        alpha=0.4,
    )
    colors = [collection.get_facecolor() for collection in collections]

    if alternatives:
        for alternative in alternatives.alternatives[:alternatives_count]:
            if alternative.category_index is None:
                color = "black"
            else:
                color = colors[alternative.category_index]
            ax.plot(
                xs, normalize_profile(problem.criteria, alternative.profile),
                "o--",
                label=alternative.name,
                color=color,
                alpha=1,
            )

    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["worst", "best"])

    for y in [0, 1]:
        ax.axhline(y=y, color="black", alpha=0.2)
    for x in xs:
        ax.axvline(x=x, color="black", alpha=0.2)

    fig.savefig(out, format="png", dpi=100)
    plt.close(fig)


def normalize_profile(criteria, ys):
    return [
        normalize_value(criterion, y)
        for (criterion, y) in zip(criteria, ys)
    ]

def normalize_value(criterion, y):
    assert criterion.is_real
    y = (y - criterion.real_min_value) / (criterion.real_max_value - criterion.real_min_value)
    if criterion.is_increasing:
        return y
    else:
        return 1 - y
