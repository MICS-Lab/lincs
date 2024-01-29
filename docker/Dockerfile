FROM python:3.10-slim AS dist
RUN mkdir /dist
COPY lincs-1.1.0a3.dev0-cp310-cp310-manylinux_2_31_x86_64.whl /dist

FROM python:3.10-slim
RUN --mount=type=bind,from=dist,source=/dist,target=/dist pip install --find-links /dist --only-binary lincs lincs==1.1.0a3.dev0
