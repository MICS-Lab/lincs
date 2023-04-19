from __future__ import annotations

import click

from . import hello as plad_hello


@click.group()
def main():
    pass


@main.command()
@click.argument("name")
def hello(name: str):
    print(plad_hello(name))
