import re

GLOBAL_META_FILE_NAME = 'global_meta.bin'
MODEL_CKPT_FILE_NAME = 'model.bin'
OPTIM_CKPT_FILE_NAME = 'optim.bin'
META_CKPT_FILE_NAME = 'meta.bin'
OTHER_CKPT_FILE_NAME = 'other.bin'

CKPT_PAT = re.compile(r'global_meta|model|optim|meta|other')
