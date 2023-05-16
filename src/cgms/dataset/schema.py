#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Tyler H. S. Lee
#
# This file is part of THL-CGMS Package.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, visit <http://www.gnu.org/licenses/>.
import pandas as pd

from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Self


class Schema(BaseModel):
    id: str
    dt: Optional[datetime]

    def combine_dt(self) -> datetime:
        NotImplementedError

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.json())


class CGMSData(Schema):
    date: str
    time: str
    glucose: float

    def combine_dt(self) -> Self:
        self.dt = datetime.combine(
            datetime.strptime(self.date, "%Y-%m-%d %H:%M:%S"),
            datetime.strptime(self.time, "%H:%M:%S").time()
        )
        return self
    

class RawMealDataV1(Schema):
    pass


class RawActivityDataV1(Schema):
    pass
    