"""The upstream-side configuration this example is compared against.

``nvidia/cosmos-framework`` publishes an action-policy DROID recipe for Nano only
(``configs/base/experiment/action/posttrain_config/`` holds four files, all ``*_nano.py``), while this
example trains Edge. Comparing the two therefore needs an upstream run that is Edge: upstream's own
framework and its Nano DROID recipe, with Edge's model config and the Edge base checkpoint. This file is
that overlay. It copies no upstream code -- it imports the recipe and the Edge model baseline and states
the three deltas -- so what the comparison runs on the upstream side stays upstream's, and what we changed
to get there is reviewable in one screen.

The checkpoint both sides start from is ``nvidia/Cosmos3-Edge``, the base. The recipe's
``keys_to_skip_loading`` is left exactly as upstream wrote it, so the action head is drawn fresh here as
well -- but each side draws from its own RNG, so one run exports what it drew and the other loads it
(``dump_action_head.py`` here, ``train_sft.load_action_head`` there). Without that the action term of the
loss compares two different models.

Place it where hydra can see it and run upstream's trainer with ``experiment=action_policy_droid_edge``::

    DROID_ROOT=/path/to/droid_plus_lerobot_640x360_20260412 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Edge DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=1 -m cosmos_framework.scripts.train experiment=action_policy_droid_edge

The deltas, and what stays Nano's, are tabulated in README.md next to this file.
"""

import copy

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from hydra.core.config_store import ConfigStore

action_policy_droid_edge = copy.deepcopy(action_policy_droid_nano)

# Delta 1: the model. Edge's baseline instead of Nano's, then the two post-construction tweaks the Nano
# recipe applies to its own copy -- they are properties of the recipe (a 33-frame window, no cap on the
# packed sequence), not of the tier, so they have to be re-applied to this fresh copy.
action_policy_droid_edge["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
action_policy_droid_edge["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]
action_policy_droid_edge["model"]["config"]["max_num_tokens_after_packing"] = -1
# The Nano recipe raises loss_scale from the Nano baseline's 1.0 to 10.0. Edge's baseline is already 10.0,
# so this assignment changes nothing; it is written out so that a reader can see the recipe's intent is
# preserved rather than silently relying on the tier default.
action_policy_droid_edge["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0

# Delta 2: the text processor comes from the checkpoint being loaded, not from a separately pinned repo.
# Edge's model baseline pins ``build_processor_lazy(repository="nvidia/Cosmos3-Edge", revision=...)`` at a
# revision that need not be the one the weights come from, and this example reads its tokenizer from the
# checkpoint directory itself. Comparing two runs that tokenize the prompt with different files would put a
# difference in the text segment before the model is even reached. The revision is pinned rather than
# "main" so a later re-tokenization cannot change what the comparison measured. (Measured: the text
# tokenizer is byte-identical between nvidia/Cosmos3-Edge and nvidia/Cosmos3-Edge-Policy-DROID -- only the
# image and video preprocessor configs differ, and those only in key order -- so this pin is about keeping
# the two sides provably on one file, not about a difference we found.)
_PROCESSOR = action_policy_droid_edge["model"]["config"]["vlm_config"]["tokenizer"]
_PROCESSOR["repository"] = "nvidia/Cosmos3-Edge"
_PROCESSOR["revision"] = "344d602b128d1bbdacb43b08d0a3626f46343e29"

# Delta 3: the job name, so runs are told apart in the logs.
action_policy_droid_edge["job"]["name"] = "action_policy_droid_edge"

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_edge",
    node=action_policy_droid_edge,
)
