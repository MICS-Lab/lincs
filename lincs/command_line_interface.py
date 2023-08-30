# Copyright 2023 Vincent Jacques

from __future__ import annotations
import math
import os
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
                # \b prevents rewrapping of the paragraph by Click
                # (https://click.palletsprojects.com/en/8.1.x/api/#click.wrap_text)
                conditions__ = f"\n\n\b\nOnly valid if:\n"
                if os.environ.get("LINCS_GENERATING_SPHINX_DOC") == "1":
                    conditions__ += "\n"
                    quote = "``"
                else:
                    quote = "'"
                for (k, v) in conditions_:
                    conditions__ += f"- {quote}{k}{quote} is {quote}{v}{quote}\n"
                # This non-blocking space ensures Click does not put the default
                # value on the same line as the last condition
                conditions__ += f"\n\n "

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
                    assert help.endswith(".") or help.endswith(")")
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
        print()
        print(command.get_help(ctx=click.Context(info_name=" ".join(["lincs"] + prefix), command=command)))
        print()

        if isinstance(command, click.Group):
            for name, command in command.commands.items():
                walk(prefix + [name], command)

    walk([], main)

    def walk(path, node, type_name=None):
        if '.'.join(path) in [
            "lincs.command_line_interface",
            "lincs.visualization.plt",
        ]:
            return

        title = f"{'.'.join(path)}: {type_name or type(node).__name__}"
        print(title)
        print("-" * len(title))

        if type_name is not None:
            return

        if type(node) in [str, bool, dict, property]:
            print()
            return

        if node.__doc__:
            print(node.__doc__)
        print()

        if type(node) in [type(walk), type(lincs.Model.dump)]:
            return

        if '.'.join(path) in [
            "lincs.Criterion.CategoryCorrelation",
            "lincs.Criterion.ValueType",
            "lincs.SufficientCoalitions.Kind",
        ]:
            for name in node.names:
                walk(path + [name], name, type_name="value")
            return

        for name in sorted(dir(node)):
            if name not in ["__init__"] and name.startswith("__") and name.endswith("__"):
                continue

            # Avoid weird recursion
            if name in path:
                continue

            walk(path + [name], getattr(node, name))

    walk(["lincs"], lincs)


@main.group(
    help="Get information about lincs itself."
)
def info():
    pass

@info.command(
    help="Check whether lincs was compiled with CUDA support. Return code is 0 if CUDA is supported, 1 otherwise."
)
@click.option(
    "--quiet", is_flag=True,
    help="Don't print anything, just return the exit code.",
)
def has_gpu(quiet):
    if lincs.has_gpu:
        if not quiet:
            print("lincs was compiled with CUDA support")
    else:
        if not quiet:
            print("lincs was compiled WITHOUT CUDA support")
        exit(1)


@main.group(
    help="Generate synthetic data.",
)
def generate():
    pass


@generate.command(
    help="""
        Generate a synthetic classification problem.

        The generated problem has CRITERIA_COUNT criteria and CATEGORIES_COUNT categories.
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
    "--denormalized-min-max",
    is_flag=True,
    help="Generate criteria with random denormalized min and max values. (By default, min and max value are 0 and 1)",
)
@click.option(
    "--allow-decreasing-criteria",
    is_flag=True,
    help="Allow criteria to have decreasing correlation to categories. (By default, all criteria have growing correlation)",
)
@click.option(
    "--output-problem",
    type=click.File(mode="w"),
    default="-",
    help="Write generated problem to this file instead of standard output.",
)
@click.option(
    "--random-seed",
    help="The random seed to use.",
    type=click.IntRange(min=0),
    default=random.randrange(2**30),
)
def classification_problem(
    criteria_count,
    categories_count,
    denormalized_min_max,
    allow_decreasing_criteria,
    output_problem,
    random_seed
):
    command_line = ["lincs", "generate", "classification-problem", criteria_count, categories_count, "--random-seed", random_seed]
    if denormalized_min_max:
        command_line += ["--denormalized-min-max"]
    if allow_decreasing_criteria:
        command_line += ["--allow-decreasing-criteria"]
    print(f"# Reproduction command: {' '.join(str(c) for c in command_line)}", file=output_problem, flush=True)

    problem = lincs.generate_classification_problem(
        criteria_count,
        categories_count,
        random_seed=random_seed,
        normalized_min_max=not denormalized_min_max,
        allow_decreasing_criteria=allow_decreasing_criteria,
    )
    problem.dump(output_problem)


@generate.command(
    help="""
        Generate a synthetic classification model.

        PROBLEM is a *classification problem* file describing the problem to generate a model for.
    """,
)
@click.argument(
    "problem",
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
    problem,
    output_model,
    random_seed,
    model_type,
    mrsort__fixed_weights_sum,
):
    command_line = ["lincs", "generate", "classification-model", get_input_file_name(problem), "--random-seed", random_seed, "--model-type", model_type]
    if mrsort__fixed_weights_sum is not None:
        command_line += ["--mrsort.fixed-weights-sum", mrsort__fixed_weights_sum]
    print(f"# Reproduction command: {' '.join(str(c) for c in command_line)}", file=output_model, flush=True)

    problem = lincs.Problem.load(problem)
    assert model_type == "mrsort"
    model = lincs.generate_mrsort_classification_model(
        problem,
        random_seed=random_seed,
        fixed_weights_sum=mrsort__fixed_weights_sum,
    )
    model.dump(problem, output_model)


def get_input_file_name(input_file):
    if input_file.name == "<stdin>":
        return "-"
    return input_file.name


@generate.command(
    help="""
        Generate synthetic classified alternatives.

        PROBLEM is a *classification problem* file describing the problem to generate alternatives for.
        MODEL is a *classification model* file for that problem describing the model to use to classify the generated alternatives.
    """,
)
@click.argument(
    "problem",
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
    "--misclassified-count",
    type=click.IntRange(min=0),
    default=0,
    help="Misclassify that many alternatives.",
)
@click.option(
    "--random-seed",
    help="The random seed to use.",
    type=click.IntRange(min=0),
    default=random.randrange(2**30),
)
def classified_alternatives(
    problem,
    model,
    alternatives_count,
    output_classified_alternatives,
    max_imbalance,
    misclassified_count,
    random_seed,
):
    command_line = ["lincs", "generate", "classified-alternatives", get_input_file_name(problem), get_input_file_name(model), alternatives_count, "--random--seed", random_seed]
    if max_imbalance is not None:
        command_line += ["--max-imbalance", max_imbalance]
    command_line += ["--misclassified-count", misclassified_count]
    print(f"# Reproduction command: {' '.join(str(c) for c in command_line)}", file=output_classified_alternatives, flush=True)

    problem = lincs.Problem.load(problem)
    model = lincs.Model.load(problem, model)
    alternatives = lincs.generate_classified_alternatives(
        problem,
        model,
        alternatives_count,
        random_seed=random_seed,
        max_imbalance=max_imbalance,
    )
    if misclassified_count:
        lincs.misclassify_alternatives(
            problem,
            alternatives,
            misclassified_count,
            random_seed=random_seed + 27,  # Arbitrary, does not hurt
        )
    alternatives.dump(problem, output_classified_alternatives)


@main.group(
    help="Make graphs from data.",
)
def visualize():
    pass


@visualize.command(
    help="""
        Visualize a classification model.

        PROBLEM is a *classification problem* file.
        MODEL is a *classification model* file for that problem describing the model to visualize.
        The generated image is written to the OUTPUT file in PNG format.
    """,
)
@click.argument(
    "problem",
    type=click.File(mode="r"),
)
@click.argument(
    "model",
    type=click.File(mode="r"),
)
@click.option(
    "--alternatives",
    type=click.File(mode="r"),
    help="Add the alternatives from this *classified alternatives* file to the visualization.",
)
@click.option(
    "--alternatives-count",
    type=int,
    help="Add only this number of alternatives.",
)
@click.argument(
    "output",
    type=click.File(mode="wb"),
)
def classification_model(
    problem,
    model,
    alternatives,
    alternatives_count,
    output,
):
    problem = lincs.Problem.load(problem)
    model = lincs.Model.load(problem, model)
    if alternatives is not None:
        alternatives = lincs.Alternatives.load(problem, alternatives)
    lincs.visualization.visualize_model(problem, model, alternatives, alternatives_count, output)


@main.group(
    help="Learn a model.",
)
def learn():
    pass


@learn.command(
    help="""
        Learn a classification model.

        PROBLEM is a *classification problem* file describing the problem to learn a model for.
        LEARNING_SET is a *classified alternatives* file for that problem.
        It's used as a source of truth to learn the model.
    """,
)
@click.argument(
    "problem",
    type=click.File(mode="r"),
)
@click.argument(
    "learning-set",
    type=click.File(mode="r"),
)
@click.option(
    "--output-model",
    type=click.File(mode="w"),
    default="-",
    help="Write the learned classification model to this file instead of standard output.",
)
@options_tree(
    "model-type",
    dict(
        help="The type of classification model to learn.",
        type=click.Choice(["mrsort", "ucncs"]),
        default="mrsort",
        show_default=True,
    ),
    {
        "mrsort": [
            (
                "strategy",
                dict(
                    help="The top-level strategy to use to learn the MRSort model. See https://mics-lab.github.io/lincs/user-guide.html#learning-strategies about strategies.",
                    type=click.Choice(["weights-profiles-breed"]),
                    default="weights-profiles-breed",
                    show_default=True,
                ),
                {
                    "weights-profiles-breed": [
                        (
                            "target-accuracy",
                            dict(
                                help="The target accuracy to reach on the learning set.",
                                type=click.FloatRange(min=0.0, max=1.0),
                                default=1.0,
                                show_default=True,
                            ),
                            {},
                        ),
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
                            "max-duration-seconds",
                            dict(
                                help="The maximum duration to learn the MRSort model, in seconds.",
                                type=click.FloatRange(min=0),
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
                                help="The strategy to use to initialize the MRSort models.",
                                type=click.Choice(["maximize-discrimination-per-criterion"]),
                                default="maximize-discrimination-per-criterion",
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "weights-strategy",
                            dict(
                                help="The strategy to use to improve the weights of the MRSort models.",
                                type=click.Choice(["linear-program"]),
                                default="linear-program",
                                show_default=True,
                            ),
                            {
                                "linear-program": [
                                    (
                                        "solver",
                                        dict(
                                            help="The solver to use to solve the linear programs.",
                                            type=click.Choice(["glop", "alglib"]),
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
                                help="The strategy to use to improve the profiles of the MRSort models.",
                                type=click.Choice(["accuracy-heuristic"]),
                                default="accuracy-heuristic",
                                show_default=True,
                            ),
                            {
                                "accuracy-heuristic": [
                                    (
                                        "random-seed",
                                        dict(
                                            help="The random seed to use for this heuristic.",
                                            type=click.IntRange(min=0),
                                            default=random.randrange(2**30),
                                        ),
                                        {},
                                    ),
                                    (
                                        "processor",
                                        dict(
                                            help="The processor to use to improve the profiles of the MRSort models."
                                            + ("" if lincs.has_gpu else " (Only 'cpu' is available because lincs was compiled without 'nvcc')"),
                                            type=click.Choice(["cpu"] + (["gpu"] if lincs.has_gpu else [])),
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
                                help="The strategy to use to breed the MRSort models.",
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
                        (
                            "verbose",
                            dict(
                                help="Print information about the learning process on stderr while learning.",
                                is_flag=True,
                            ),
                            {},
                        )
                    ],
                },
            ),
        ],
        "ucncs": [
            (
                "approach",
                dict(
                    help="The general approach to transform the learning problem into a satisfiability problem.",
                    type=click.Choice(["sat-by-coalitions", "sat-by-separation", "max-sat-by-coalitions", "max-sat-by-separation"]),
                    default="sat-by-coalitions",
                    show_default=True,
                ),
                {},
            ),
        ],
    },
)
def classification_model(
    problem,
    learning_set,
    output_model,
    model_type,
    mrsort__strategy,
    mrsort__weights_profiles_breed__target_accuracy,
    mrsort__weights_profiles_breed__max_iterations,
    mrsort__weights_profiles_breed__max_duration_seconds,
    mrsort__weights_profiles_breed__models_count,
    mrsort__weights_profiles_breed__initialization_strategy,
    mrsort__weights_profiles_breed__weights_strategy,
    mrsort__weights_profiles_breed__linear_program__solver,
    mrsort__weights_profiles_breed__profiles_strategy,
    mrsort__weights_profiles_breed__accuracy_heuristic__random_seed,
    mrsort__weights_profiles_breed__accuracy_heuristic__processor,
    mrsort__weights_profiles_breed__breed_strategy,
    mrsort__weights_profiles_breed__reinitialize_least_accurate__portion,
    mrsort__weights_profiles_breed__verbose,
    ucncs__approach,
):
    command_line = ["lincs", "learn", "classification-model", get_input_file_name(problem), get_input_file_name(learning_set), "--model-type", model_type]

    problem = lincs.Problem.load(problem)
    learning_set = lincs.Alternatives.load(problem, learning_set)

    if model_type == "mrsort":
        command_line += ["--mrsort.strategy", mrsort__strategy]
        if mrsort__strategy == "weights-profiles-breed":
            command_line += ["--mrsort.weights-profiles-breed.models-count", mrsort__weights_profiles_breed__models_count]
            command_line += ["--mrsort.weights-profiles-breed.accuracy-heuristic.random-seed", mrsort__weights_profiles_breed__accuracy_heuristic__random_seed]

            learning_data = lincs.LearnMrsortByWeightsProfilesBreed.LearningData.make(problem, learning_set, mrsort__weights_profiles_breed__models_count, mrsort__weights_profiles_breed__accuracy_heuristic__random_seed)

            command_line += ["--mrsort.weights-profiles-breed.initialization-strategy", mrsort__weights_profiles_breed__initialization_strategy]
            if mrsort__weights_profiles_breed__initialization_strategy == "maximize-discrimination-per-criterion":
                profiles_initialization_strategy = lincs.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.weights-strategy", mrsort__weights_profiles_breed__weights_strategy]
            if mrsort__weights_profiles_breed__weights_strategy == "linear-program":
                command_line += ["--mrsort.weights-profiles-breed.linear-program.solver", mrsort__weights_profiles_breed__linear_program__solver]
                if mrsort__weights_profiles_breed__linear_program__solver == "glop":
                    weights_optimization_strategy = lincs.OptimizeWeightsUsingGlop(learning_data)
                elif mrsort__weights_profiles_breed__linear_program__solver == "alglib":
                    weights_optimization_strategy = lincs.OptimizeWeightsUsingAlglib(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.profiles-strategy", mrsort__weights_profiles_breed__profiles_strategy]
            if mrsort__weights_profiles_breed__profiles_strategy == "accuracy-heuristic":
                command_line += ["--mrsort.weights-profiles-breed.accuracy-heuristic.processor", mrsort__weights_profiles_breed__accuracy_heuristic__processor]
                if mrsort__weights_profiles_breed__accuracy_heuristic__processor == "cpu":
                    profiles_improvement_strategy = lincs.ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
                elif mrsort__weights_profiles_breed__accuracy_heuristic__processor == "gpu":
                    assert lincs.has_gpu
                    profiles_improvement_strategy = lincs.ImproveProfilesWithAccuracyHeuristicOnGpu(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.breed-strategy", mrsort__weights_profiles_breed__breed_strategy]
            if mrsort__weights_profiles_breed__breed_strategy == "reinitialize-least-accurate":
                command_line += ["--mrsort.weights-profiles-breed.reinitialize-least-accurate.portion", mrsort__weights_profiles_breed__reinitialize_least_accurate__portion]
                count = int(mrsort__weights_profiles_breed__reinitialize_least_accurate__portion * mrsort__weights_profiles_breed__models_count)
                breeding_strategy = lincs.ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, count)

            command_line += ["--mrsort.weights-profiles-breed.target-accuracy", mrsort__weights_profiles_breed__target_accuracy]
            termination_strategies = [lincs.TerminateAtAccuracy(
                learning_data,
                math.ceil(mrsort__weights_profiles_breed__target_accuracy * len(learning_set.alternatives)),
            )]
            if mrsort__weights_profiles_breed__max_iterations is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-iterations", mrsort__weights_profiles_breed__max_iterations]
                termination_strategies.append(lincs.TerminateAfterIterations(learning_data, mrsort__weights_profiles_breed__max_iterations))
            if mrsort__weights_profiles_breed__max_duration_seconds is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-duration-seconds", mrsort__weights_profiles_breed__max_duration_seconds]
                termination_strategies.append(lincs.TerminateAfterSeconds(mrsort__weights_profiles_breed__max_duration_seconds))
            if len(termination_strategies) == 1:
                termination_strategy = termination_strategies[0]
            else:
                termination_strategy = lincs.TerminateWhenAny(termination_strategies)

            observers = []
            if mrsort__weights_profiles_breed__verbose:
                class VerboseObserver(lincs.LearnMrsortByWeightsProfilesBreed.Observer):
                    def __init__(self, learning_data):
                        super().__init__()
                        self.learning_data = learning_data

                    def after_iteration(self):
                        print("Best accuracy:", self.learning_data.get_best_accuracy(), file=sys.stderr)

                    def before_return(self):
                        print("Final accuracy:", self.learning_data.get_best_accuracy(), file=sys.stderr)

                observers.append(VerboseObserver(learning_data))

            learning = lincs.LearnMrsortByWeightsProfilesBreed(
                learning_data,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
                observers,
            )
    elif model_type == "ucncs":
        command_line += ["--ucncs.approach", ucncs__approach]
        if ucncs__approach == "sat-by-coalitions":
            learning = lincs.LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set)
        elif ucncs__approach == "sat-by-separation":
            learning = lincs.LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set)
        elif ucncs__approach == "max-sat-by-coalitions":
            learning = lincs.LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat(problem, learning_set)
        elif ucncs__approach == "max-sat-by-separation":
            learning = lincs.LearnUcncsByMaxSatBySeparationUsingEvalmaxsat(problem, learning_set)

    print(f"# Reproduction command: {' '.join(str(c) for c in command_line)}", file=output_model, flush=True)

    model = learning.perform()
    model.dump(problem, output_model)


@main.command(
    help="""
        Classify alternatives.

        PROBLEM is a *classification problem* file.
        MODEL is a *classification model* file for that problem.
        ALTERNATIVES is an *unclassified alternatives* file for that problem.
    """,
)
@click.argument(
    "problem",
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
    problem,
    model,
    alternatives,
    output_classified_alternatives,
):
    problem = lincs.Problem.load(problem)
    model = lincs.Model.load(problem, model)
    alternatives = lincs.Alternatives.load(problem, alternatives)
    lincs.classify_alternatives(problem, model, alternatives)
    alternatives.dump(problem, output_classified_alternatives)


@main.command(
    help="""
        Compute a classification accuracy.

        PROBLEM is a *classification problem* file.
        MODEL is a *classification model* file for that problem.
        TESTING_SET is a *classified alternatives* file for that problem.

        The classification accuracy is written to standard output as an integer between 0 and the number of alternatives.
    """,
)
@click.argument(
    "problem",
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
    problem,
    model,
    testing_set,
):
    problem = lincs.Problem.load(problem)
    model = lincs.Model.load(problem, model)
    testing_set = lincs.Alternatives.load(problem, testing_set)
    result = lincs.classify_alternatives(problem, model, testing_set)
    print(f"{result.unchanged}/{result.changed + result.unchanged}")
