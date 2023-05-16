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
import pandas as pd
from cgms.dataset import Dataset


def test_data_load(load_data: Dataset):
    cgms = load_data

    rc = pd.read_excel(pytest.cgms_file_path)
    assert len(cgms.stack) == len(rc)
