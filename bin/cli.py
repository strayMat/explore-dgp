#! /usr/bin/env python
import logging

import click

from explore_dgp.constants import LOG_LEVEL
from explore_dgp.utils import hello

logging.basicConfig(
    level=logging.getLevelName(LOG_LEVEL),
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)


@click.group()
@click.version_option(package_name="explore_dgp")
def cli():
    pass


@cli.command()
def main() -> None:
    """explore_dgp Main entrypoint"""
    click.secho(hello(), fg="green")


if __name__ == "__main__":
    cli()
