# Copyright 2023 Vincent Jacques

import matplotlib.pyplot as plt


def visualize_model(problem, model, alternatives, axes: plt.Axes):
    xs = [criterion.name for criterion in problem.criteria]
    boundary_profiles = [[] for _ in problem.ordered_categories[1:]]
    for criterion_index, criterion in enumerate(problem.criteria):
        acc_vals = model.accepted_values[criterion_index]
        if criterion.is_real:
            for boundary_index in range(len(problem.ordered_categories) - 1):
                boundary_profiles[boundary_index].append(acc_vals.real_thresholds[boundary_index])
        elif criterion.is_integer:
            for boundary_index in range(len(problem.ordered_categories) - 1):
                boundary_profiles[boundary_index].append(acc_vals.integer_thresholds[boundary_index])
        elif criterion.is_enumerated:
            for boundary_index in range(len(problem.ordered_categories) - 1):
                boundary_profiles[boundary_index].append(criterion.get_value_rank(acc_vals.enumerated_thresholds[boundary_index]))
        else:
            assert False
    ys = [
        normalize_profile(problem.criteria, boundary_profile)
        for boundary_profile in boundary_profiles
    ]
    ys.append([1] * len(xs))
    unstacked_ys = [ys[0]]
    for ys1, ys2 in zip(ys[1:], ys[:-1]):
        unstacked_ys.append([y1 - y2 for y1, y2 in zip(ys1, ys2)])
    collections = axes.stackplot(
        xs, unstacked_ys,
        labels=[category.name for category in problem.ordered_categories],
        alpha=0.4,
    )
    colors = [collection.get_facecolor() for collection in collections]

    if alternatives:
        for alternative in alternatives:
            if alternative.category_index is None:
                color = "black"
            else:
                color = colors[alternative.category_index]
            profile = []
            for criterion_index, criterion in enumerate(problem.criteria):
                if criterion.is_real:
                    profile.append(alternative.profile[criterion_index].real_value)
                elif criterion.is_integer:
                    profile.append(alternative.profile[criterion_index].integer_value)
                elif criterion.is_enumerated:
                    profile.append(criterion.get_value_rank(alternative.profile[criterion_index].enumerated_value))
                else:
                    assert False
            axes.plot(
                xs, normalize_profile(problem.criteria, profile),
                "o--",
                label=alternative.name,
                color=color,
                alpha=1,
            )

    axes.legend()
    axes.set_ylim(-0.05, 1.05)
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["worst", "best"])

    for y in [0, 1]:
        axes.axhline(y=y, color="black", alpha=0.2)
    for x in xs:
        axes.axvline(x=x, color="black", alpha=0.2)


def normalize_profile(criteria, ys):
    return [
        normalize_value(criterion, y)
        for (criterion, y) in zip(criteria, ys)
    ]


def normalize_value(criterion, y):
    if criterion.is_real:
        y = (y - criterion.real_min_value) / (criterion.real_max_value - criterion.real_min_value)
    elif criterion.is_integer:
        y = (y - criterion.integer_min_value) / (criterion.integer_max_value - criterion.integer_min_value)
    elif criterion.is_enumerated:
        y = y / (len(criterion.ordered_values) - 1)
    else:
        assert False

    if criterion.is_enumerated or criterion.is_increasing:
        return y
    else:
        return 1 - y
