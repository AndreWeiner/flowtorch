"""Protocol-only worker used to test the pvpython subprocess client."""

import json
import os
import sys

import numpy as np

from flowtorch.data._tecplot_pvpython import _send_array, _send_error, _send_json


def main():
    stream = os.fdopen(int(sys.argv[1]), "wb", buffering=0)
    _send_json(stream, {"protocol": 1})
    for line in sys.stdin.buffer:
        request = json.loads(line)
        operation = request["operation"]
        if operation == "close":
            _send_json(stream, None)
            break
        if operation == "json":
            _send_json(stream, request["value"])
        elif operation == "array":
            _send_array(stream, np.arange(12, dtype=np.float64).reshape(3, 4), np)
        else:
            _send_error(stream, ValueError("fake worker error"))
    stream.close()


if __name__ == "__main__":
    main()
