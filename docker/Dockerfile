FROM python:3.10-slim AS dist
RUN mkdir /dist
ARG lincs_version
COPY lincs-$lincs_version-cp310-cp310-manylinux_2_31_x86_64.whl /dist

FROM python:3.10-slim
ARG lincs_version
RUN --mount=type=bind,from=dist,source=/dist,target=/dist pip install --find-links /dist --only-binary lincs lincs==$lincs_version
