import matplotlib.pyplot as plt


def visualize_model(domain, model, alternatives, out):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")

    xs = [criterion.name for criterion in domain.criteria]
    for boundary_index, boundary in enumerate(model.boundaries):
        label = f"{domain.categories[boundary_index].name} and {domain.categories[boundary_index + 1].name}"
        ax.plot(xs, list(boundary.profile), "o-", label=label)

    if alternatives:
        for alternative in alternatives.alternatives:
            ax.plot(xs, list(alternative.profile), "o--", label=alternative.name)

    ax.legend()
    ax.set_ylim(0, 1)

    fig.savefig(out, format="png", dpi=100)
    plt.close(fig)
