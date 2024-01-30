<!-- WARNING: this file is generated from 'doc-sources/python-api/python-api.ipynb'. MANUAL EDITS WILL BE LOST. -->

# The Python API

This document builds up on {doc}`our "Get Started" guide <get-started>` and our {doc}`user guide <user-guide>`, and introduces *lincs*' Python API.
This API is more flexible, albeit more complex, than the command-line interface you've been using so far.

## Do it again, in Python

First, lets do exactly the same thing as in our "Get started" guide, but using the Python API.


```python
from lincs import classification as lc
```

Generate a synthetic classification problem:


```python
problem = lc.generate_problem(criteria_count=4, categories_count=3, random_seed=40)
```

The first difference with the command-line interface is the third argument to the call to `generate_problem`: it's the pseudo-random seed optionally passed by the `--random-seed` option on the command line. All pseudo-random seeds are mandatory in the Python API, so that you have full control of reproducibility. If you don't care about it, you can use `random.randrange(2**30)` to use pseudo-random pseudo-random seeds. (No typo here: the pseudo-random seeds are pseudo-random.)

Generated problems are returned as Python objects of class `lincs.Problem`. You can print them:


```python
import sys
problem.dump(sys.stdout)
```

```yaml
kind: classification-problem
format_version: 1
criteria:
  - name: Criterion 1
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
  - name: Criterion 2
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
  - name: Criterion 3
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
  - name: Criterion 4
    value_type: real
    preference_direction: increasing
    min_value: 0
    max_value: 1
ordered_categories:
  - name: Worst category
  - name: Intermediate category 1
  - name: Best category
```


Description functions generate a list of strings:


```python
print("\n".join(lc.describe_problem(problem)))
```

```text
This a classification problem into 3 ordered categories named "Worst category", "Intermediate category 1" and "Best category".
The best category is "Best category" and the worst category is "Worst category".
There are 4 classification criteria (in no particular order).
Criterion "Criterion 1" takes real values between 0.0 and 1.0 included.
Higher values of "Criterion 1" are known to be better.
Criterion "Criterion 2" takes real values between 0.0 and 1.0 included.
Higher values of "Criterion 2" are known to be better.
Criterion "Criterion 3" takes real values between 0.0 and 1.0 included.
Higher values of "Criterion 3" are known to be better.
Criterion "Criterion 4" takes real values between 0.0 and 1.0 included.
Higher values of "Criterion 4" are known to be better.
```


Generate a synthetic MR-Sort classification model, again with an explicit pseudo-random seed:


```python
model = lc.generate_mrsort_model(problem, random_seed=41)

model.dump(problem, sys.stdout)
```

```yaml
kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.255905151, 0.676961303]
  - kind: thresholds
    thresholds: [0.0551739037, 0.324553937]
  - kind: thresholds
    thresholds: [0.162252158, 0.673279881]
  - kind: thresholds
    thresholds: [0.0526000932, 0.598555863]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.147771254, 0.618687689, 0.406786472, 0.0960085914]
  - *coalitions
```


Visualization functions interface with [Matplotlib](https://matplotlib.org/):


```python
import matplotlib.pyplot as plt
```


```python
axes = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")[1]
lc.visualize_model(problem, model, [], axes)
```



![png](python-api_files/python-api_15_0.png)



Get the model's description:


```python
print("\n".join(lc.describe_model(problem, model)))
```

```text
This is a MR-Sort (a.k.a. 1-Uc-NCS) model: an NCS model where the sufficient coalitions are specified using the same criterion weights for all boundaries.
The weights associated to each criterion are:
  - Criterion "Criterion 1": 0.15
  - Criterion "Criterion 2": 0.62
  - Criterion "Criterion 3": 0.41
  - Criterion "Criterion 4": 0.10
To get into an upper category, an alternative must be better than the following profiles on a set of criteria whose weights add up to at least 1:
  - For category "Intermediate category 1": at least 0.26 on criterion "Criterion 1", at least 0.06 on criterion "Criterion 2", at least 0.16 on criterion "Criterion 3", and at least 0.05 on criterion "Criterion 4"
  - For category "Best category": at least 0.68 on criterion "Criterion 1", at least 0.32 on criterion "Criterion 2", at least 0.67 on criterion "Criterion 3", and at least 0.60 on criterion "Criterion 4"
```


Generate a synthetic learning set (with an explicit pseudo-random seed):


```python
learning_set = lc.generate_classified_alternatives(problem, model, alternatives_count=1000, random_seed=42)
```

Dump it (in memory instead of on `sys.stdout` to print only the first few lines):


```python
import io
f = io.StringIO()
learning_set.dump(problem, f)
print("\n".join(f.getvalue().splitlines()[:6]))
```

```text
name,"Criterion 1","Criterion 2","Criterion 3","Criterion 4",category
"Alternative 1",0.37454012,0.796543002,0.95071429,0.183434784,"Best category"
"Alternative 2",0.731993914,0.779690981,0.598658502,0.596850157,"Intermediate category 1"
"Alternative 3",0.156018645,0.445832759,0.15599452,0.0999749228,"Worst category"
"Alternative 4",0.0580836125,0.4592489,0.866176128,0.333708614,"Best category"
"Alternative 5",0.601114988,0.14286682,0.708072603,0.650888503,"Intermediate category 1"
```


Visualize it:


```python
axes = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")[1]
lc.visualize_model(problem, model, learning_set.alternatives[:5], axes)
```



![png](python-api_files/python-api_23_0.png)



Let's now train a new model from this synthetic learning set.
The command-line interface of `lincs learn classification-model` accepts quite a few options.
Most of them set up the strategies used for the learning, as described further in our {doc}`user guide <user-guide>`.
When using the Python API, you have to create these strategies yourself:


```python
learning_data = lc.LearnMrsortByWeightsProfilesBreed.LearningData(problem, learning_set, models_count=9, random_seed=43)
profiles_initialization_strategy = lc.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(learning_data)
weights_optimization_strategy = lc.OptimizeWeightsUsingGlop(learning_data)
profiles_improvement_strategy = lc.ImproveProfilesWithAccuracyHeuristicOnCpu(learning_data)
breeding_strategy = lc.ReinitializeLeastAccurate(learning_data, profiles_initialization_strategy=profiles_initialization_strategy, count=4)
termination_strategy = lc.TerminateAtAccuracy(learning_data, target_accuracy=len(learning_set.alternatives))
```

Then create the learning itself:


```python
learning = lc.LearnMrsortByWeightsProfilesBreed(
    learning_data,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    breeding_strategy,
    termination_strategy,
)
```

And `.perform` it to create the learned `Model` object:


```python
learned_model = learning.perform()
learned_model.dump(problem, sys.stdout)
```

```yaml
kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.339874953, 0.421424538]
  - kind: thresholds
    thresholds: [0.0556534864, 0.326433569]
  - kind: thresholds
    thresholds: [0.162616938, 0.67343241]
  - kind: thresholds
    thresholds: [0.0878681168, 0.252649099]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0, 1.01327896e-06, 0.999998987, 0]
  - *coalitions
```


Create a testing set and classify it, taking notes of the accuracy of the new model on that testing set:


```python
testing_set = lc.generate_classified_alternatives(problem, model, alternatives_count=3000, random_seed=44)
classification_result = lc.classify_alternatives(problem, learned_model, testing_set)
classification_result.changed, classification_result.unchanged
```




```text
(4, 2996)
```



This covers what was done in our "Get started" guide.
As you can see the Python API is more verbose, but for good reasons: it's more powerful as you'll see in the next section.

## Do more, with the Python API

### Create classification objects

You don't have to use our pseudo-random generation functions; you can create `Problem`, `Model`, *etc.* instances yourself.

#### Create a `Problem`


```python
problem = lc.Problem(
    criteria=[
        lc.Criterion("Physics grade", lc.Criterion.IntegerValues(lc.Criterion.PreferenceDirection.increasing, 0, 100)),
        lc.Criterion("Literature grade", lc.Criterion.EnumeratedValues(["f", "e", "d", "c", "b", "a"])),
    ],
    categories=[lc.Category("Failed"), lc.Category("Passed"), lc.Category("Congratulations")],
)

problem.dump(sys.stdout)
```

```yaml
kind: classification-problem
format_version: 1
criteria:
  - name: Physics grade
    value_type: integer
    preference_direction: increasing
    min_value: 0
    max_value: 100
  - name: Literature grade
    value_type: enumerated
    ordered_values: [f, e, d, c, b, a]
ordered_categories:
  - name: Failed
  - name: Passed
  - name: Congratulations
```


You can access all their attributes in code as well:


```python
criterion = problem.criteria[0]
```


```python
criterion.name
```




```text
'Physics grade'
```




```python
criterion.value_type, criterion.is_real, criterion.is_integer, criterion.is_enumerated
```




```text
(liblincs.ValueType.integer, False, True, False)
```




```python
values = criterion.integer_values
```


```python
values.preference_direction, values.is_increasing, values.is_decreasing
```




```text
(liblincs.PreferenceDirection.isotone, True, False)
```




```python
values.min_value, values.max_value
```




```text
(0, 100)
```




```python
criterion = problem.criteria[1]
```


```python
criterion.name
```




```text
'Literature grade'
```




```python
criterion.value_type, criterion.is_real, criterion.is_integer, criterion.is_enumerated
```




```text
(liblincs.ValueType.enumerated, False, False, True)
```




```python
values = criterion.enumerated_values
```


```python
list(values.ordered_values)
```




```text
['f', 'e', 'd', 'c', 'b', 'a']
```




```python
values.get_value_rank(value="a")
```




```text
5
```



#### Create a `Model`


```python
model = lc.Model(
    problem,
    accepted_values=[
        lc.AcceptedValues(lc.AcceptedValues.IntegerThresholds([50, 80])),
        lc.AcceptedValues(lc.AcceptedValues.EnumeratedThresholds(["c", "a"])),
    ],
    sufficient_coalitions=[
        lc.SufficientCoalitions(lc.SufficientCoalitions.Weights([0.5, 0.5])),
        lc.SufficientCoalitions(lc.SufficientCoalitions.Weights([0.5, 0.5])),
    ],
)

model.dump(problem, sys.stdout)
```

```yaml
kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [50, 80]
  - kind: thresholds
    thresholds: [c, a]
sufficient_coalitions:
  - &coalitions
    kind: weights
    criterion_weights: [0.5, 0.5]
  - *coalitions
```



```python
accepted = model.accepted_values[0]
```


```python
accepted.value_type, accepted.is_real, accepted.is_integer, accepted.is_enumerated
```




```text
(liblincs.ValueType.integer, False, True, False)
```




```python
accepted.kind, accepted.is_thresholds
```




```text
(liblincs.Kind.thresholds, True)
```




```python
list(accepted.integer_thresholds.thresholds)
```




```text
[50, 80]
```




```python
accepted = model.accepted_values[1]
```


```python
accepted.value_type, accepted.is_real, accepted.is_integer, accepted.is_enumerated
```




```text
(liblincs.ValueType.enumerated, False, False, True)
```




```python
accepted.kind, accepted.is_thresholds
```




```text
(liblincs.Kind.thresholds, True)
```




```python
list(accepted.enumerated_thresholds.thresholds)
```




```text
['c', 'a']
```




```python
sufficient = model.sufficient_coalitions[0]
```


```python
sufficient.kind, sufficient.is_weights, sufficient.is_roots
```




```text
(liblincs.Kind.weights, True, False)
```




```python
list(sufficient.weights.criterion_weights)
```




```text
[0.5, 0.5]
```



#### Create (classified) `Alternatives`


```python
alternatives = lc.Alternatives(problem, [
    lc.Alternative(
        "Unclassified alternative",
        [
            lc.Performance(lc.Performance.Integer(50)),
            lc.Performance(lc.Performance.Enumerated("c")),
        ],
        None
    ),
    lc.Alternative(
        "Classified alternative",
        [
            lc.Performance(lc.Performance.Integer(90)),
            lc.Performance(lc.Performance.Enumerated("a")),
        ],
        2
    ),
])
```


```python
alternative = alternatives.alternatives[0]
```


```python
alternative.category_index is None
```




```text
True
```




```python
performance = alternative.profile[0]
```


```python
performance.value_type, performance.is_real, performance.is_integer, performance.is_enumerated
```




```text
(liblincs.ValueType.integer, False, True, False)
```




```python
performance.integer.value
```




```text
50
```




```python
problem.ordered_categories[alternatives.alternatives[1].category_index].name
```




```text
'Congratulations'
```



### Clone classification objects

Just use [`copy.deepcopy`](https://docs.python.org/3/library/copy.html#copy.deepcopy):


```python
import copy

copied_problem = copy.deepcopy(problem)
copied_model = copy.deepcopy(model)
copied_alternatives = copy.deepcopy(alternatives)
```

This is especially useful *e.g.* if you want to identify alternatives that are classified differently by two models, because `lc.classify_alternatives` mutates the alternatives: clone the `Alternatives`, classify the copy and iterate over the [`zip`](https://docs.python.org/3/library/functions.html#zip) of both `Alternatives`, comparing their `.category_index`.

### Serialize classification objects

#### In YAML and CSV like the command-line

(and the upcomming C++ API)

Classification objects have a `.dump` method, and their classes have a static `.load` method that accept file-like objects.

We've used them above to print classification objects to `sys.stdout`. Here is an example of how to use them with actual files:


```python
with open("problem.yml", "w") as f:
    problem.dump(f)

with open("model.yml", "w") as f:
    model.dump(problem, f)

with open("alternatives.csv", "w") as f:
    alternatives.dump(problem, f)

with open("problem.yml") as f:
    problem = lc.Problem.load(f)

with open("model.yml") as f:
    model = lc.Model.load(problem, f)

with open("alternatives.csv") as f:
    alternatives = lc.Alternatives.load(problem, f)
```

And here with in-memory [io](https://docs.python.org/3/library/io.html) objects:


```python
f = io.StringIO()
problem.dump(f)
s = f.getvalue()
print(s)
```

```yaml
kind: classification-problem
format_version: 1
criteria:
  - name: Physics grade
    value_type: integer
    preference_direction: increasing
    min_value: 0
    max_value: 100
  - name: Literature grade
    value_type: enumerated
    ordered_values: [f, e, d, c, b, a]
ordered_categories:
  - name: Failed
  - name: Passed
  - name: Congratulations
```




```python
f = io.StringIO(s)
problem = lc.Problem.load(f)
```

#### Using the Python-specific `pickle` module

Classification objects simply support [pickling](https://docs.python.org/3/library/pickle.html) and unpickling. We recommend using the YAML and CSV formats whenever possible because they are not tied to the Python language (or the *lincs* library for that matter).


```python
import pickle

pickle.loads(pickle.dumps(problem)).dump(sys.stdout)
```

```yaml
kind: classification-problem
format_version: 1
criteria:
  - name: Physics grade
    value_type: integer
    preference_direction: increasing
    min_value: 0
    max_value: 100
  - name: Literature grade
    value_type: enumerated
    ordered_values: [f, e, d, c, b, a]
ordered_categories:
  - name: Failed
  - name: Passed
  - name: Congratulations
```

