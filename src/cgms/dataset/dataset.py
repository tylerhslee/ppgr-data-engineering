#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Tyler H. S. Lee
#
# This file is part of THL-CGMS Package.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, visit <http://www.gnu.org/licenses/>.
'''Load & validate input data from XLSX files.'''
from __future__ import annotations

import pandas as pd
from enum import auto, Enum
from pathlib import Path
from pydantic import BaseModel
from typing import List, Self

from .schema import Schema


class FileFormat(Enum):
    XLSX = auto()


class Dataset(BaseModel):
    '''Dataset used in CGMS prediction model.'''
    file_path: Path
    file_format: FileFormat
    stack: List[Schema] = []

    def load(self, schema: Schema, **kwargs) -> Self:
        if self.file_format == FileFormat.XLSX:
            df = pd.read_excel(
                self.file_path,
                dtype=str,
                **kwargs
            )

        for row in df.to_dict('records'):
            self.stack.append(schema(**{
                k: v for k, v in row.items()
                    if k in schema.__fields__
            }).combine_dt())
        
        return self
 