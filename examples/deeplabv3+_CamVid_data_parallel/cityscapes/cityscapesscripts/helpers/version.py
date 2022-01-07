#!/usr/bin/env python

import os

with open(os.path.join(os.path.dirname(__file__), '..', 'VERSION')) as f:
    version = f.read().strip()

if __name__ == "__main__":
    print(version)
