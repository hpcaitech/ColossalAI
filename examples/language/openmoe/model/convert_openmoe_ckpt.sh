python ColossalAI/examples/language/openmoe/model/convert_openmoe_ckpt.py \
--t5x_checkpoint_path checkpoint_553000 \
--config_file ColossalAI/examples/language/openmoe/model/openmoe_8b_config.json \
--pytorch_dump_path openmoe_8b_chat_ckpt \
--target_dtype float32 \
--lazy