#!/bin/bash
set -xe

cd amp_with_booster && bash ./test_ci.sh && cd ..
cd gradient_accumulation && bash ./test_ci.sh && cd ..
cd gradient_clipping && bash ./test_ci.sh && cd ..
