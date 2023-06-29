# Copyright 2023 Vincent Jacques

ARG PYTHON_VERSION=3.11

FROM python:$PYTHON_VERSION-slim AS dist
COPY dist /dist

FROM python:$PYTHON_VERSION-slim

ARG LINCS_VERSION
RUN --mount=type=bind,from=dist,source=/dist,target=/dist pip3 install --find-links /dist --only-binary lincs lincs==$LINCS_VERSION
