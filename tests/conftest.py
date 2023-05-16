#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Tyler H. S. Lee
#
# This file is part of THL-CGMS Package.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, visit <http://www.gnu.org/licenses/>.
import pytest

from cgms.dataset import Dataset, FileFormat
from cgms.dataset.schema import CGMSData


def pytest_configure():
    pytest.cgms_file_path = "data/raw/cgms timeseries data.xlsx"


@pytest.fixture(scope="session")
def load_data():
    cgms = Dataset(
        file_path = pytest.cgms_file_path,
        file_format = FileFormat.XLSX
    ).load(schema=CGMSData)

    return cgms


