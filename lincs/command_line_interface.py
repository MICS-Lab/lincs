# Copyright 2023-2024 Vincent Jacques

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import sys

import click
import matplotlib.pyplot as plt

import lincs


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
@click.version_option(version=lincs.__version__, message="%(version)s")
def main():
    pass


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
    "--forbid-increasing-criteria",
    is_flag=True,
    help="Forbid criteria to have increasing preference direction. (Requires '--allow-decreasing-criteria')",
)
@click.option(
    "--allow-decreasing-criteria",
    is_flag=True,
    help="Allow criteria to have decreasing preference direction. (By default, all criteria have increasing preference direction)",
)
@click.option(
    "--forbid-real-criteria",
    is_flag=True,
    help="Forbid criteria with real values. (Requires another '--allow-...-criteria' option)",
)
@click.option(
    "--allow-enumerated-criteria",
    is_flag=True,
    help="Allow criteria with enumerated values. (By default, all criteria are real)",
)
@click.option(
    "--allow-integer-criteria",
    is_flag=True,
    help="Allow criteria with integer values. (By default, all criteria are real)",
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
    forbid_increasing_criteria, allow_decreasing_criteria,
    forbid_real_criteria, allow_enumerated_criteria, allow_integer_criteria,
    output_problem,
    random_seed
):
    command_line = ["lincs", "generate", "classification-problem", criteria_count, categories_count, "--random-seed", random_seed]
    if denormalized_min_max:
        command_line += ["--denormalized-min-max"]

    allowed_preference_directions = []
    if forbid_increasing_criteria:
        command_line += ["--forbid-increasing-criteria"]
    else:
        allowed_preference_directions.append(lincs.classification.Criterion.PreferenceDirection.increasing)
    if allow_decreasing_criteria:
        command_line += ["--allow-decreasing-criteria"]
        allowed_preference_directions.append(lincs.classification.Criterion.PreferenceDirection.decreasing)

    allowed_value_types = []
    if forbid_real_criteria:
        command_line += ["--forbid-real-criteria"]
    else:
        allowed_value_types.append(lincs.classification.Criterion.ValueType.real)
    if allow_enumerated_criteria:
        command_line += ["--allow-enumerated-criteria"]
        allowed_value_types.append(lincs.classification.Criterion.ValueType.enumerated)
    if allow_integer_criteria:
        command_line += ["--allow-integer-criteria"]
        allowed_value_types.append(lincs.classification.Criterion.ValueType.integer)

    if not allowed_value_types:
        print("ERROR: no allowed value type. '--forbid-real-criteria' requires at least one of '--allow-enumerated-criteria' or '--allow-integer-criteria'", file=sys.stderr)
        print(make_reproduction_command(command_line), file=sys.stderr)
        exit(1)

    if not allowed_preference_directions:
        print("ERROR: no allowed preference direction. '--forbid-increasing-criteria' requires '--allow-decreasing-criteria'", file=sys.stderr)
        print(make_reproduction_command(command_line), file=sys.stderr)
        exit(1)

    print(f"# {make_reproduction_command(command_line)}", file=output_problem, flush=True)

    problem = lincs.classification.generate_problem(
        criteria_count,
        categories_count,
        random_seed=random_seed,
        normalized_min_max=not denormalized_min_max,
        allowed_preference_directions=allowed_preference_directions,
        allowed_value_types=allowed_value_types,
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
    print(f"# {make_reproduction_command(command_line)}", file=output_model, flush=True)

    with loading_guard():
        problem = lincs.classification.Problem.load(problem)

    assert model_type == "mrsort"
    model = lincs.classification.generate_mrsort_model(
        problem,
        random_seed=random_seed,
        fixed_weights_sum=mrsort__fixed_weights_sum,
    )
    model.dump(problem, output_model)


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
    "--output-alternatives",
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
# @todo(Feature, later) Consider creating a 'lincs misclassify-alternatives' command
# that mimics the 'lincs generate classified-alternatives --misclassified-count' option below,
# but on a pre-existing learning set
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
    output_alternatives,
    max_imbalance,
    misclassified_count,
    random_seed,
):
    command_line = ["lincs", "generate", "classified-alternatives", get_input_file_name(problem), get_input_file_name(model), alternatives_count, "--random-seed", random_seed]
    if max_imbalance is not None:
        command_line += ["--max-imbalance", max_imbalance]
    command_line += ["--misclassified-count", misclassified_count]

    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        model = lincs.classification.Model.load(problem, model)

    try:
        alternatives = lincs.classification.generate_alternatives(
            problem,
            model,
            alternatives_count,
            random_seed=random_seed,
            max_imbalance=max_imbalance,
        )
    except lincs.classification.BalancedAlternativesGenerationException:
        print("ERROR: lincs is unable to generate a balanced set of classified alternatives. Try to increase the allowed imbalance, or use a more lenient model?", file=sys.stderr)
        print(make_reproduction_command(command_line), file=sys.stderr)
        exit(1)
    else:
        if misclassified_count:
            lincs.classification.misclassify_alternatives(
                problem,
                alternatives,
                misclassified_count,
                random_seed=random_seed + 27,  # Arbitrary, does not hurt
            )
        print(f"# {make_reproduction_command(command_line)}", file=output_alternatives, flush=True)
        alternatives.dump(problem, output_alternatives)


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
    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        model = lincs.classification.Model.load(problem, model)
        if alternatives is None:
            alternatives = []
        else:
            alternatives = lincs.classification.Alternatives.load(problem, alternatives).alternatives
            if alternatives_count is not None:
                alternatives = alternatives[:alternatives_count]

    figure, axes = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    lincs.classification.visualize_model(problem, model, alternatives, axes)
    figure.savefig(output, format="png", dpi=100)
    plt.close(figure)


@main.group(
    help="Provide human-readable descriptions.",
)
def describe():
    pass


@describe.command(
    help="""
        Describe a classification problem.

        PROBLEM is a *classification problem* file.
    """,
)
@click.argument(
    "problem",
    type=click.File(mode="r"),
)
@click.option(
    "--output-description",
    type=click.File(mode="w"),
    default="-",
    help="Write description to this file instead of standard output.",
)
def classification_problem(
    problem,
    output_description,
):
    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
    for line in lincs.classification.describe_problem(problem):
        print(line, file=output_description)


@describe.command(
    help="""
        Describe a classification model.

        PROBLEM is a *classification problem* file.
        MODEL is a *classification model* file for that problem.
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
    "--output-description",
    type=click.File(mode="w"),
    default="-",
    help="Write description to this file instead of standard output.",
)
def classification_model(
    problem,
    model,
    output_description,
):
    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        model = lincs.classification.Model.load(problem, model)
    for line in lincs.classification.describe_model(problem, model):
        print(line, file=output_description)


@main.group(
    help="Learn a model.",
)
def learn():
    pass


max_sat_options = [
    (
        "solver",
        dict(
            help="The solver to use to solve the MaxSAT problem.",
            type=click.Choice(["eval-max-sat"]),
            default="eval-max-sat",
            show_default=True,
        ),
        {
            "eval-max-sat": [
                # These three options correspond to EvalMaxSAT's command-line options here:
                # https://github.com/normal-account/EvalMaxSAT2022/blob/main/main.cpp#L43-L50
                (
                    "nb-minimize-threads",
                    dict(
                        help="The number of threads to use to minimize the MaxSAT problem. Passed directly to the EvalMaxSAT solver.",
                        type=click.IntRange(min=0),
                        default=0,
                        show_default=True,
                    ),
                    {},
                ),
                (
                    "timeout-fast-minimize",
                    dict(
                        help="The maximum duration of the \"fast minimize\" phase of solving the MaxSAT problem, in seconds. Passed directly to the EvalMaxSAT solver.",
                        type=click.IntRange(min=0),
                        default=60,
                        show_default=True,
                    ),
                    {},
                ),
                (
                    "coef-minimize-time",
                    dict(
                        help="The coefficient to use to multiply the time spent minimizing the MaxSAT problem. Passed directly to the EvalMaxSAT solver.",
                        type=click.IntRange(min=0),
                        default=2,
                        show_default=True,
                    ),
                    {},
                ),
            ],
        },
    ),
]

@learn.command(
    help="""
        Learn a classification model.

        PROBLEM is a *classification problem* file describing the problem to learn a model for.
        LEARNING_SET is a *classified alternatives* file for that problem.
        It's used as a source of truth to learn the model.

        If you use the --mrsort.weights-profiles-breed strategy, you SHOULD specify at least one
        termination strategy, e.g. --mrsort.weights-profiles-breed.max-duration.
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
                            "max-iterations-without-progress",
                            dict(
                                help="The maximum number of iterations to try learning the MRSort model without progressing before giving up.",
                                type=click.IntRange(min=1),
                                default=None,
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "max-duration",
                            dict(
                                help="The maximum duration to learn the MRSort model, in seconds.",
                                type=click.FloatRange(min=0),
                                default=None,
                                show_default=True,
                            ),
                            {},
                        ),
                        (
                            "max-duration-without-progress",
                            dict(
                                help="The maximum duration to try learning the MRSort model without progressing before giving up, in seconds.",
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
                        ),
                        (
                            "output-metadata",
                            dict(
                                help="Write metadata about the learning process to this file.",
                                type=click.File(mode="w"),
                            ),
                            {},
                        )
                    ],
                },
            ),
        ],
        "ucncs": [
            (
                "strategy",
                dict(
                    help="The general approach to transform the learning problem into a satisfiability problem.",
                    type=click.Choice(["sat-by-coalitions", "sat-by-separation", "max-sat-by-coalitions", "max-sat-by-separation"]),
                    default="sat-by-coalitions",
                    show_default=True,
                ),
                {
                    "max-sat-by-coalitions": max_sat_options,
                    "max-sat-by-separation": max_sat_options,
                },
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
    mrsort__weights_profiles_breed__max_iterations_without_progress,
    mrsort__weights_profiles_breed__max_duration,
    mrsort__weights_profiles_breed__max_duration_without_progress,
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
    mrsort__weights_profiles_breed__output_metadata,
    ucncs__strategy,
    ucncs__max_sat_by_coalitions__solver,
    ucncs__max_sat_by_coalitions__eval_max_sat__nb_minimize_threads,
    ucncs__max_sat_by_coalitions__eval_max_sat__timeout_fast_minimize,
    ucncs__max_sat_by_coalitions__eval_max_sat__coef_minimize_time,
    ucncs__max_sat_by_separation__solver,
    ucncs__max_sat_by_separation__eval_max_sat__nb_minimize_threads,
    ucncs__max_sat_by_separation__eval_max_sat__timeout_fast_minimize,
    ucncs__max_sat_by_separation__eval_max_sat__coef_minimize_time,
):
    command_line = ["lincs", "learn", "classification-model", get_input_file_name(problem), get_input_file_name(learning_set), "--model-type", model_type]

    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        learning_set = lincs.classification.Alternatives.load(problem, learning_set)

    if model_type == "mrsort":
        command_line += ["--mrsort.strategy", mrsort__strategy]
        if mrsort__strategy == "weights-profiles-breed":
            command_line += ["--mrsort.weights-profiles-breed.models-count", mrsort__weights_profiles_breed__models_count]
            command_line += ["--mrsort.weights-profiles-breed.accuracy-heuristic.random-seed", mrsort__weights_profiles_breed__accuracy_heuristic__random_seed]

            learning_data = lincs.classification.LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, mrsort__weights_profiles_breed__models_count, mrsort__weights_profiles_breed__accuracy_heuristic__random_seed)

            command_line += ["--mrsort.weights-profiles-breed.initialization-strategy", mrsort__weights_profiles_breed__initialization_strategy]
            if mrsort__weights_profiles_breed__initialization_strategy == "maximize-discrimination-per-criterion":
                profiles_initialization_strategy = lincs.classification.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.weights-strategy", mrsort__weights_profiles_breed__weights_strategy]
            if mrsort__weights_profiles_breed__weights_strategy == "linear-program":
                command_line += ["--mrsort.weights-profiles-breed.linear-program.solver", mrsort__weights_profiles_breed__linear_program__solver]
                if mrsort__weights_profiles_breed__linear_program__solver == "glop":
                    weights_optimization_strategy = lincs.classification.OptimizeWeightsUsingGlop(learning_data)
                elif mrsort__weights_profiles_breed__linear_program__solver == "alglib":
                    weights_optimization_strategy = lincs.classification.OptimizeWeightsUsingAlglib(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.profiles-strategy", mrsort__weights_profiles_breed__profiles_strategy]
            if mrsort__weights_profiles_breed__profiles_strategy == "accuracy-heuristic":
                command_line += ["--mrsort.weights-profiles-breed.accuracy-heuristic.processor", mrsort__weights_profiles_breed__accuracy_heuristic__processor]
                if mrsort__weights_profiles_breed__accuracy_heuristic__processor == "cpu":
                    profiles_improvement_strategy = lincs.classification.ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
                elif mrsort__weights_profiles_breed__accuracy_heuristic__processor == "gpu":
                    assert lincs.has_gpu
                    profiles_improvement_strategy = lincs.classification.ImproveProfilesWithAccuracyHeuristicOnGpu(learning_data)

            command_line += ["--mrsort.weights-profiles-breed.breed-strategy", mrsort__weights_profiles_breed__breed_strategy]
            if mrsort__weights_profiles_breed__breed_strategy == "reinitialize-least-accurate":
                command_line += ["--mrsort.weights-profiles-breed.reinitialize-least-accurate.portion", mrsort__weights_profiles_breed__reinitialize_least_accurate__portion]
                count = int(mrsort__weights_profiles_breed__reinitialize_least_accurate__portion * mrsort__weights_profiles_breed__models_count)
                breeding_strategy = lincs.classification.ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy, count)

            command_line += ["--mrsort.weights-profiles-breed.target-accuracy", mrsort__weights_profiles_breed__target_accuracy]
            termination_strategies = [lincs.classification.TerminateAtAccuracy(
                learning_data,
                math.ceil(mrsort__weights_profiles_breed__target_accuracy * len(learning_set.alternatives)),
            )]
            if mrsort__weights_profiles_breed__max_iterations is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-iterations", mrsort__weights_profiles_breed__max_iterations]
                termination_strategies.append(lincs.classification.TerminateAfterIterations(learning_data, mrsort__weights_profiles_breed__max_iterations))
            if mrsort__weights_profiles_breed__max_iterations_without_progress is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-iterations-without-progress", mrsort__weights_profiles_breed__max_iterations_without_progress]
                termination_strategies.append(lincs.classification.TerminateAfterIterationsWithoutProgress(learning_data, mrsort__weights_profiles_breed__max_iterations_without_progress))
            if mrsort__weights_profiles_breed__max_duration is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-duration", mrsort__weights_profiles_breed__max_duration]
                termination_strategies.append(lincs.classification.TerminateAfterSeconds(mrsort__weights_profiles_breed__max_duration))
            if mrsort__weights_profiles_breed__max_duration_without_progress is not None:
                command_line += ["--mrsort.weights-profiles-breed.max-duration-without-progress", mrsort__weights_profiles_breed__max_duration_without_progress]
                termination_strategies.append(lincs.classification.TerminateAfterSecondsWithoutProgress(mrsort__weights_profiles_breed__max_duration_without_progress))
            if len(termination_strategies) == 1:
                termination_strategy = termination_strategies[0]
            else:
                termination_strategy = lincs.classification.TerminateWhenAny(termination_strategies)

            observers = []
            if mrsort__weights_profiles_breed__verbose:
                class VerboseObserver(lincs.classification.LearnMrsortByWeightsProfilesBreed.Observer):
                    def __init__(self, learning_data):
                        super().__init__()
                        self.learning_data = learning_data

                    def after_iteration(self):
                        print(f"Best accuracy (after {self.learning_data.iteration_index + 1} iterations): {self.learning_data.get_best_accuracy()}", file=sys.stderr)

                    def before_return(self):
                        print(f"Final accuracy (after {self.learning_data.iteration_index + 1} iterations): {self.learning_data.get_best_accuracy()}", file=sys.stderr)

                observers.append(VerboseObserver(learning_data))
            if mrsort__weights_profiles_breed__output_metadata:
                class MetadataObserver(lincs.classification.LearnMrsortByWeightsProfilesBreed.Observer):
                    def __init__(self, learning_data):
                        super().__init__()
                        self.learning_data = learning_data
                        self.accuracies = []

                    def after_iteration(self):
                        self.accuracies.append(self.learning_data.get_best_accuracy())

                    def before_return(self):
                        self.accuracies.append(self.learning_data.get_best_accuracy())

                metadata_observer = MetadataObserver(learning_data)
                observers.append(metadata_observer)

            learning = lincs.classification.LearnMrsortByWeightsProfilesBreed(
                learning_data,
                profiles_initialization_strategy,
                weights_optimization_strategy,
                profiles_improvement_strategy,
                breeding_strategy,
                termination_strategy,
                observers,
            )
    elif model_type == "ucncs":
        command_line += ["--ucncs.strategy", ucncs__strategy]
        if ucncs__strategy == "sat-by-coalitions":
            learning = lincs.classification.LearnUcncsBySatByCoalitionsUsingMinisat(problem, learning_set)
        elif ucncs__strategy == "sat-by-separation":
            learning = lincs.classification.LearnUcncsBySatBySeparationUsingMinisat(problem, learning_set)
        elif ucncs__strategy == "max-sat-by-coalitions":
            command_line += ["--ucncs.max-sat-by-coalitions.solver", ucncs__max_sat_by_coalitions__solver]
            if ucncs__max_sat_by_coalitions__solver == "eval-max-sat":
                command_line += [
                    "--ucncs.max-sat-by-coalitions.eval-max-sat.nb-minimize-threads", ucncs__max_sat_by_coalitions__eval_max_sat__nb_minimize_threads,
                    "--ucncs.max-sat-by-coalitions.eval-max-sat.timeout-fast-minimize", ucncs__max_sat_by_coalitions__eval_max_sat__timeout_fast_minimize,
                    "--ucncs.max-sat-by-coalitions.eval-max-sat.coef-minimize-time", ucncs__max_sat_by_coalitions__eval_max_sat__coef_minimize_time,
                ]
                learning = lincs.classification.LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat(
                    problem,
                    learning_set,
                    ucncs__max_sat_by_coalitions__eval_max_sat__nb_minimize_threads,
                    ucncs__max_sat_by_coalitions__eval_max_sat__timeout_fast_minimize,
                    ucncs__max_sat_by_coalitions__eval_max_sat__coef_minimize_time,
                )
        elif ucncs__strategy == "max-sat-by-separation":
            command_line += ["--ucncs.max-sat-by-separation.solver", ucncs__max_sat_by_separation__solver]
            if ucncs__max_sat_by_separation__solver == "eval-max-sat":
                command_line += [
                    "--ucncs.max-sat-by-separation.eval-max-sat.nb-minimize-threads", ucncs__max_sat_by_separation__eval_max_sat__nb_minimize_threads,
                    "--ucncs.max-sat-by-separation.eval-max-sat.timeout-fast-minimize", ucncs__max_sat_by_separation__eval_max_sat__timeout_fast_minimize,
                    "--ucncs.max-sat-by-separation.eval-max-sat.coef-minimize-time", ucncs__max_sat_by_separation__eval_max_sat__coef_minimize_time,
                ]
                learning = lincs.classification.LearnUcncsByMaxSatBySeparationUsingEvalmaxsat(
                    problem,
                    learning_set,
                    ucncs__max_sat_by_separation__eval_max_sat__nb_minimize_threads,
                    ucncs__max_sat_by_separation__eval_max_sat__timeout_fast_minimize,
                    ucncs__max_sat_by_separation__eval_max_sat__coef_minimize_time,
                )

    try:
        model = learning.perform()
    except lincs.LearningFailureException:
        print("ERROR: lincs is unable to learn from this learning set using this algorithm and these parameters.", file=sys.stderr)
        print(make_reproduction_command(command_line), file=sys.stderr)
        exit(1)
    else:
        print(f"# {make_reproduction_command(command_line)}", file=output_model, flush=True)
        if model_type == "mrsort" and mrsort__strategy == "weights-profiles-breed" and mrsort__weights_profiles_breed__output_metadata is not None:
            for termination_strategy in termination_strategies:
                if termination_strategy.terminate():
                    termination_condition = {
                        lincs.classification.TerminateAtAccuracy: "target accuracy reached",
                        lincs.classification.TerminateAfterIterations: "maximum total number of iterations reached",
                        lincs.classification.TerminateAfterIterationsWithoutProgress: "maximum number of iterations without progress reached",
                        lincs.classification.TerminateAfterSeconds: "maximum total duration reached",
                        lincs.classification.TerminateAfterSecondsWithoutProgress: "maximum duration without progress reached",
                    }.get(
                        termination_strategy.__class__,
                        f"{termination_strategy.__class__.__name__} (Unexpected, please let the lincs maintainers know about this)"
                    )
                    break
            else:
                termination_condition = "unknown (Unexpected, please let the lincs maintainers know about this)"
            json.dump(
                {
                    "termination_condition": termination_condition,
                    "iterations_count": learning_data.iteration_index,
                    "intermediate_accuracies": metadata_observer.accuracies,
                },
                mrsort__weights_profiles_breed__output_metadata,
                indent=4
            )
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
    "--output-alternatives",
    type=click.File(mode="w"),
    default="-",
    help="Write classified alternatives to this file instead of standard output.",
)
def classify(
    problem,
    model,
    alternatives,
    output_alternatives,
):
    command_line = ["lincs", "classify", get_input_file_name(problem), get_input_file_name(model), get_input_file_name(alternatives)]

    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        model = lincs.classification.Model.load(problem, model)
        alternatives = lincs.classification.Alternatives.load(problem, alternatives)

    lincs.classification.classify_alternatives(problem, model, alternatives)
    print(f"# {make_reproduction_command(command_line)}", file=output_alternatives, flush=True)
    alternatives.dump(problem, output_alternatives)


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
    with loading_guard():
        problem = lincs.classification.Problem.load(problem)
        model = lincs.classification.Model.load(problem, model)
        testing_set = lincs.classification.Alternatives.load(problem, testing_set)

    result = lincs.classification.classify_alternatives(problem, model, testing_set)
    print(f"{result.unchanged}/{result.changed + result.unchanged}")


@contextlib.contextmanager
def loading_guard():
    try:
        yield
    except lincs.DataValidationException as e:
        print(f"ERROR: lincs found an issue with the input files: {e}", file=sys.stderr)
        exit(1)


def get_input_file_name(input_file):
    if input_file.name == "<stdin>":
        return "-"
    return input_file.name


def make_reproduction_command(command):
    return f"Reproduction command (with lincs version {lincs.__version__}): {' '.join(str(c) for c in command)}"
