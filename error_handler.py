"""
    A state-less command-line tool that trains linear regression models.
    Copyright 2019 Solomon Victorino

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys

from package_metadata import Package


class ErrorHandler:
    @staticmethod
    def command_failed(ex: Exception):
        print("Fatal error: ", + ex)
        print("The command you issued can't be fulfilled. This likely isn't your fault. Please report a bug at " +
              Package.BUG_REPORT_URL + ".")
        sys.exit(-1)
