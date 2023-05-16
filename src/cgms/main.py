#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Tyler H. S. Lee
#
# This file is part of THL-CGMS Package.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, visit <http://www.gnu.org/licenses/>.
import typer

app = typer.Typer()


@app.callback()
def main():
    typer.secho(
        message = "Continuous Glucose Monitoring System (CGMS) Data Package"
    )


@app.command()
def ping():
    typer.secho(
        message = "Ping success!",
        # color = typer.colors.BRIGHT_GREEN
    )
    return 1
