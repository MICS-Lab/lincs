from __future__ import annotations
import random
import subprocess
import sys

import click

import lincs
import lincs.visualization


def options_tree(name, kwds, dependents):
    """
    Ad-hoc decorator for a tree dependent of click.options.
    Options down the tree are only valid if the options up the tree are set to specific values.
    Think of them as sub-options of the previous ones.
    """
    def decorator(command):
        def walk(
            option_name_prefix_,
            parameter_name_prefix_,
            conditions_,
            name_,
            kwds_,
            dependents_,
        ):
            if conditions_:
                conditions__ = "\n\n\b\nOnly valid if:\n"
                for (k, v) in conditions_:
                    conditions__ += f" - '{k}' is '{v}'\n"
                conditions__ += "\n\n*"

            for value__, dependents__ in dependents_.items():
                for name___, kwds___, dependents___ in reversed(dependents__):
                    walk(
                        f"{option_name_prefix_}{value__}.",
                        f"{parameter_name_prefix_}{value__.replace('-', '_')}__",
                        conditions_ + [(f"--{option_name_prefix_}{name_}", value__)],
                        name___,
                        kwds___,
                        dependents___,
                    )

            help = kwds_.pop("help", None)
            if conditions_:
                if help is None:
                    help = conditions__
                else:
                    assert help.endswith(".")
                    help = f"{help} {conditions__}"

            click.option(
                f"--{option_name_prefix_}{name_}",
                f"{parameter_name_prefix_}{name_.replace('-', '_')}",
                help=help,
                **kwds_
            )(
                command,
            )

        walk("", "", [], name, kwds, dependents)

        return command

    return decorator


@click.group(
    help="""
        lincs (Learn and Infer Non-Compensatory Sorting) is a set of tools for training and using MCDA models.
    """,
)
def main():
    pass


@main.command(
    hidden=True,
)
def help_all():
    def walk(prefix, command):
        if command.hidden:
            return

        title = f"lincs {' '.join(prefix)}".rstrip()
        print(title)
        print("=" * len(title))
        print(flush=True)
        subprocess.run(["lincs"] + prefix + ["--help"], check=True)
        print()

        if isinstance(command, click.Group):
            for name, command in command.commands.items():
                walk(prefix + [name], command)

    walk([], main)

    for obj_name in sorted(dir(lincs)):
        if obj_name.startswith("_"):
            continue

        obj = getattr(lincs, obj_name)
        print(f"lincs.{obj_name}")
        print("=" * len(f"lincs.{obj_name}"))
        print()
        if obj.__doc__:
            print(obj.__doc__)
            print()
        if isinstance(obj, type):
            for attr_name in sorted(dir(obj)):
                if attr_name.startswith("_"):
                    continue
                if (
                    obj_name in {"CategoryCorrelation", "SufficientCoalitionsKind", "ValueType"}
                    and
                    attr_name in {"as_integer_ratio", "bit_count", "bit_length", "conjugate", "denominator", "from_bytes", "imag", "numerator", "to_bytes", "attr_name", "name", "names", "values"}
                ):
                    continue
                if (
                    obj_name in {"CategoryCorrelation", "SufficientCoalitionsKind"}
                    and
                    attr_name in {"real"}
                ):
                    continue
                print(f"{attr_name}")
                print("-" * len(attr_name))
                print()
                doc = getattr(obj, attr_name).__doc__
                if doc:
                    print(doc)
                    print()


@main.group(
    help="Generate synthetic data.",
)
def generate():
    pass


@generate.command(
    help="""
        Generate a synthetic classification domain.

        The generated domain has CRITERIA_COUNT criteria and CATEGORIES_COUNT categories.
    """,
)
@click.argument(
    "criteria-count",
    type=click.IntRange(min=1),
)
@click.argument(
    "categories-count",
    type=click.IntRange(min=1),
)
@click.option(
    "--output-domain",
    type=click.File(mode="w"),
    default="-",
    help="Write generated domain to this file instead of standard output.",
)
@click.option(
    "--random-seed",
    help="The random seed to use.",
    type=click.IntRange(min=0),
    default=random.randrange(2**30),
)
def classification_domain(
    criteria_count,
    categories_count,
    output_domain,
    random_seed
):
    domain = lincs.generate_domain(
        criteria_count,
        categories_count,
        random_seed=random_seed,
    )
    domain.dump(output_domain)


@generate.command(
    help="""
        Generate a synthetic classification model.

        DOMAIN is a *classification domain* file describing the domain to generate a model for.
    """,
)
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.option(
    "--output-model",
    type=click.File(mode="w"),
    default="-",
    help="Write generated model to this file instead of standard output.",
)
@click.option(
    "--random-seed",
    help="The random seed to use.",
    type=click.IntRange(min=0),
    default=random.randrange(2**30),
)
@options_tree(
    "model-type",
    dict(
        help="The type of classification model to generate.",
        type=click.Choice(["mrsort"]),
        default="mrsort",
        show_default=True,
    ),
    {
        "mrsort": [
            (
                "fixed-weights-sum",
                dict(
                    help="Make sure weights add up to this pre-determined value instead of a pseudo-random one.",
                    type=click.FloatRange(min=1.0),
                    default=None,
                    show_default=True,
                ),
                {},
            ),
        ],
    },
)
def classification_model(
    domain,
    output_model,
    random_seed,
    model_type,
    mrsort__fixed_weights_sum,
):
    domain = lincs.load_domain(domain)
    assert model_type == "mrsort"
    model = lincs.generate_mrsort_model(
        domain,
        random_seed=random_seed,
        fixed_weights_sum=mrsort__fixed_weights_sum,
    )
    model.dump(output_model)


@generate.command(
    help="""
        Generate synthetic classified alternatives.

        DOMAIN is a *classification domain* file describing the domain to generate alternatives for.
        MODEL is a *classification model* file for that domain describing the model to use to classify the generated alternatives.
    """,
)
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.argument(
    "model",
    type=click.File(mode="r"),
)
@click.argument(
    "alternatives-count",
    type=click.IntRange(min=1),
)
@click.option(
    "--output-classified-alternatives",
    type=click.File(mode="w"),
    default="-",
    help="Write generated classified alternatives to this file instead of standard output.",
)
@click.option(
    "--max-imbalance",
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    default=None,
    help="Ensure that categories are balanced, by forcing their size to differ from the perfectly balanced size by at most this fraction.",
)
@click.option(
    "--random-seed",
    help="The random seed to use.",
    type=click.IntRange(min=0),
    default=random.randrange(2**30),
)
def classified_alternatives(
    domain,
    model,
    alternatives_count,
    output_classified_alternatives,
    max_imbalance,
    random_seed,
):
    domain = lincs.load_domain(domain)
    model = lincs.load_model(domain, model)
    alternatives = lincs.generate_alternatives(
        domain,
        model,
        alternatives_count,
        random_seed=random_seed,
        max_imbalance=max_imbalance,
    )
    alternatives.dump(output_classified_alternatives)


@main.group()
def visualize():
    pass


@visualize.command()
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.argument(
    "model",
    type=click.File(mode="r"),
)
@click.option(
    "--alternatives",
    type=click.File(mode="r"),
)
@click.argument(
    "output",
    type=click.File(mode="wb"),
)
def model(
    domain,
    model,
    alternatives,
    output,
):
    domain = lincs.load_domain(domain)
    model = lincs.load_model(domain, model)
    if alternatives is not None:
        alternatives = lincs.load_alternatives(domain, alternatives)
    lincs.visualization.visualize_model(domain, model, alternatives, output)


@main.group(
    help="Learn a model.",
    hidden=True,
)
def learn():
    pass


@learn.command(
    help="""
        Learn a classification model.

        DOMAIN is a *classification domain* file describing the domain to learn a model for.
        LEARNING_SET is a *classified alternatives* file for that domain.
        It's used as a source of truth to learn the model.

        The learned *classification model* file is written to OUTPUT_MODEL, which defaults to - to write to the standard output.
    """,
)
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.argument(
    "learning-set",
    type=click.File(mode="r"),
)
@click.option(
    "--target-accuracy",
    help="The target accuracy to reach on the learning set.\n\n*",
    type=click.FloatRange(min=0.0, max=1.0),
    default=1.0,
    show_default=True,
)
@click.option(
    "--max-duration-seconds",
    help="The maximum duration of the learning process in seconds.\n\n*",
    type=click.FloatRange(min=0),
    default=None,
    show_default=True,
)
@click.argument(
    "output-model",
    type=click.File(mode="w"),
    default="-",
)
@click.option(
    "--random-seed",
    help="""
        The random seed to use.

        Some learning strategies are deterministic, pseudo-random processes.
        This seed is used to initialize the pseudo-random number generator used by these strategies.

        *
    """,
    type=click.IntRange(min=0),
)
@options_tree(
    "model-type",
    dict(
        help="The type of classification model to learn.",
        type=click.Choice(["mrsort"]),
    ),
    {
        "ucncs": [],
        "mrsort": [
            (
                "strategy",
                dict(
                    help="The top-level strategy to use to learn the MRSort model. See (@todo Add link to doc) for details about the available strategies.",
                    type=click.Choice(["weights-profiles-breed"]),
                    default="weights-profiles-breed",
                    show_default=True,
                ),
                {
                    "weights-profiles-breed": [
                        (
                            "max-iterations",
                            dict(
                                help="The maximum number of iterations to use to learn the MRSort model.",
                                type=click.IntRange(min=1),
                                default=None,
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "models-count",
                            dict(
                                help="The number of temporary MRSort models to train. The result of the learning will be the most accurate of those models.",
                                type=click.IntRange(min=1),
                                default=9,
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "initialization-strategy",
                            dict(
                                help="The strategy to use to initialize the MRSort models. See (@todo Add link to doc) for details about the available strategies.",
                                type=click.Choice(["maximize-discrimination-per-criterion"]),
                                default="maximize-discrimination-per-criterion",
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "weights-strategy",
                            dict(
                                help="The strategy to use to improve the weights of the MRSort models. See (@todo Add link to doc) for details about the available strategies.",
                                type=click.Choice(["linear-program"]),
                                default="linear-program",
                                show_default=True,
                            ),
                            {
                                "linear-program": [
                                    (
                                        "solver",
                                        dict(
                                            help="The solver to use to solve the linear programs. See (@todo Add link to doc) for details of available solvers.",
                                            type=click.Choice(["glop"]),
                                            default="glop",
                                            show_default=True,
                                        ),
                                        {},
                                    ),
                                ],
                            },
                        ),
                        (
                            "profiles-strategy",
                            dict(
                                help="The strategy to use to improve the profiles of the MRSort models. See (@todo Add link to doc) for details about the available strategies.",
                                type=click.Choice(["accuracy-heuristic"]),
                                default="accuracy-heuristic",
                                show_default=True,
                            ),
                            {
                                "accuracy-heuristic": [
                                    (
                                        "processor",
                                        dict(
                                            help="The processor to use to improve the profiles of the MRSort models.",
                                            type=click.Choice(["cpu", "gpu"]),
                                            default="cpu",
                                            show_default=True,
                                        ),
                                        {},
                                    ),
                                ],
                            },
                        ),
                        (
                            "breed-strategy",
                            dict(
                                help="The strategy to use to breed the MRSort models. See (@todo Add link to doc) for details about the available strategies.",
                                type=click.Choice(["reinitialize-least-accurate"]),
                                default="reinitialize-least-accurate",
                                show_default=True,
                            ),
                            {
                                "reinitialize-least-accurate": [
                                    (
                                        "portion",
                                        dict(
                                            help="The portion of the least accurate MRSort models to reinitialize.",
                                            type=click.FloatRange(min=0.0, max=1.0),
                                            default=0.5,
                                            show_default=True,
                                        ),
                                        {},
                                    ),
                                ],
                            },
                        ),
                    ],
                },
            ),
        ],
    },
)
def classification_model(
    domain,
    learning_set,
    target_accuracy,
    max_duration_seconds,
    output_model,
    model_type,
    mrsort__strategy,
    mrsort__weights_profiles_breed__max_iterations,
    mrsort__weights_profiles_breed__models_count,
    mrsort__weights_profiles_breed__initialization_strategy,
    mrsort__weights_profiles_breed__weights_strategy,
    mrsort__weights_profiles_breed__linear_program__solver,
    mrsort__weights_profiles_breed__profiles_strategy,
    mrsort__weights_profiles_breed__accuracy_heuristic__processor,
    mrsort__weights_profiles_breed__breed_strategy,
    mrsort__weights_profiles_breed__reinitialize_least_accurate__portion,
):
    pass


@main.command(
    help="""
        Classify alternatives.

        DOMAIN is a *classification domain* file.
        MODEL is a *classification model* file for that domain.
        ALTERNATIVES is an *unclassified alternatives* file for that domain.
    """,
)
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.argument(
    "model",
    type=click.File(mode="r"),
)
@click.argument(
    "alternatives",
    type=click.File(mode="r"),
)
@click.option(
    "--output-classified-alternatives",
    type=click.File(mode="w"),
    default="-",
    help="Write classified alternatives to this file instead of standard output.",
)
def classify(
    domain,
    model,
    alternatives,
    output_classified_alternatives,
):
    domain = lincs.load_domain(domain)
    model = lincs.load_model(domain, model)
    alternatives = lincs.load_alternatives(domain, alternatives)
    lincs.classify_alternatives(domain, model, alternatives)
    alternatives.dump(output_classified_alternatives)


@main.command(
    help="""
        Compute a classification accuracy.

        DOMAIN is a *classification domain* file.
        MODEL is a *classification model* file for that domain.
        TESTING_SET is a *classified alternatives* file for that domain.

        The classification accuracy is written to standard output as an integer between 0 and the number of alternatives.
    """,
)
@click.argument(
    "domain",
    type=click.File(mode="r"),
)
@click.argument(
    "model",
    type=click.File(mode="r"),
)
@click.argument(
    "testing-set",
    type=click.File(mode="r"),
)
def classification_accuracy(
    domain,
    model,
    testing_set,
):
    domain = lincs.load_domain(domain)
    model = lincs.load_model(domain, model)
    testing_set = lincs.load_alternatives(domain, testing_set)
    result = lincs.classify_alternatives(domain, model, testing_set)
    print(f"{result.unchanged}/{result.changed + result.unchanged}")
