import sys

from package_metadata import Package


class ErrorHandler:
    @staticmethod
    def command_failed(ex: Exception):
        print("Fatal error: ", + ex)
        print("The command you issued can't be fulfilled. This likely isn't your fault. Please report a bug at " +
              Package.BUG_REPORT_URL + ".")
        sys.exit(-1)
