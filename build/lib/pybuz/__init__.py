"""pybuz - Codes for buzlab format data"""

__version__ = '0.1.0'
__author__ = 'Yuki Fujishima <yfujishima1001@gmail.com>'

import sys
import warnings
if sys.version_info[0] != 3:
    warnings.warn(
        "This package is supported for Python 3.+, "
        "you are currently using {}.{}.{}".format(
            sys.version_info[0],
            sys.version_info[1],
            sys.version_info[2]
        )
    )