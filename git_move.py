import os
import subprocess

OLD_PATH = '/home/lclsg/data2/projects/colossal/github/lsg/ColossalAI/tests/test_auto_parallel/test_tensor_shard/test_deprecated/'
NEW_PATH = '/home/lclsg/data2/projects/colossal/github/lsg/ColossalAI/tests/test_auto_parallel/test_tensor_shard/test_deprecated/'

with open('git_move.list', 'r') as f:
    for line in os.listdir(
            '/home/lclsg/data2/projects/colossal/github/lsg/ColossalAI/tests/test_auto_parallel/test_tensor_shard/test_deprecated'
    ):
        # for line in f:
        old_file_name = line.strip()

        if old_file_name.endswith('.py'):
            old_file_path = OLD_PATH + old_file_name
            new_file_name = 'test_deprecated' + old_file_name.lstrip('test')
            new_file_path = NEW_PATH + new_file_name
            # print(f'{old_file_path} -> {new_file_path}')
            subprocess.run(["git", "mv", old_file_path, new_file_path])
            # print(new_file_path)
