Tests run using `./make.sh build/tests/library/learning-optim-tests.sh.ok --assume-new library/learning-optim-tests.sh`.

# Parallelize the outer loop in `improve_profiles`

Before:

- CPU+GPU: 0m26.288s
- full CPU: 1m10.929s

After:

- CPU+GPU: 0m25.046s
- full CPU: 0m26.161s

# Parallelize the outer loop in `optimize_weights`

Before:

- CPU+GPU: 0m25.046s
- full CPU: 0m26.161s

After:

- CPU+GPU: 0m4.560s
- full CPU: 0m5.367s

# Re-use linear programs and solvers in `optimize_weights`

Before:

- CPU+GPU: 0m26.995s
- full CPU: 0m46.461s
