# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Filter and clean documents:
Capable to clean docs with less than 512 characters, less than
256 characters and contains javascript, fix text and dataset specific
cleaning like stories and realnews datasets.
Program arguments have the details.
"""

import argparse
import glob
import json
import multiprocessing
import os
import re
import time
from functools import partial
from pathlib import Path

import ftfy
from langdetect import detect


def process_doc(json_line, args):

    # Read the line.
    document = json.loads(json_line)
    text = document['text']

    output = {'remove_512': False, 'remove_256_javascript': False, \
        'remove_512_non_english': False, 'ftfy_fix_text': False, \
        'general_cleaning': False}

    try:
        # Remove all docs with less than 512 characters
        if "remove_512" in args.tasks:
            if len(text) < 512:
                output['remove_512'] = True
                return output, text, document, True

        # Remove docs if less than 256 character length and contains Javascript
        if "remove_256_javascript" in args.tasks:
            if len(text) < 256 and 'javascript' in text.lower():
                output['remove_256_javascript'] = True
                return output, text, document, True

        # Remove docs < 512 and nonenglish
        if "remove_512_non_english" in args.tasks:
            if len(text) < 512 and detect(text) != 'en':
                output['remove_512_non_english'] = True
                return output, text, document, True

        # Fix the text using ftfy, don't remove the text, hence return False
        if "ftfy_fix_text" in args.tasks:
            fixed_text = ftfy.fix_text(text)
            output['ftfy_fix_text'] = True
            return output, fixed_text, document, False

        # Cleaning extra spaces and newlines
        if "general_cleaning" in args.tasks:
            cleaned_text = re.sub(r"  +|\b\n+ |\b\n+", " ", text)
            #cleaned_text = re.sub(r"\n\n+", "\n\n", text) # used this for Gutenberg dataset
            #cleaned_text = re.sub(r"\n", "\n\n", text) # Used this for realnews

            # stories datasets
            #cleaned_text = re.sub(r" \'", "'", text)
            #cleaned_text = re.sub(r" \!", "!", cleaned_text)
            #cleaned_text = re.sub(r" \.", ".", cleaned_text)
            #cleaned_text = re.sub(r" \?", "?", cleaned_text)
            #cleaned_text = re.sub(r" - ", "-", cleaned_text)
            ##cleaned_text = re.sub(r"\" ", "\"", cleaned_text)
            #cleaned_text = re.sub(r" @ ", "@", cleaned_text)

            output['general_cleaning'] = True
            return output, cleaned_text, document, False

    except Exception as e:
        print('Error: *************************\n{}\ntext: {}'.format(e, \
            text), flush=True)
        return output, text, document, True

    # don't remove
    return output, text, document, False


def process_set(args, input_file, output_f_cleaned, output_f_filtered):

    print(' > working on {} ...'.format(input_file), flush=True)

    num_docs = num_remove_512 = num_remove_java = num_remove_512_non_english \
        = num_ftfy_fix_text = num_general_cleaning = 0

    # Output file and counters.
    output_cleaned = open(output_f_cleaned, 'wb')
    output_filtered = open(output_f_filtered, 'wb')

    start_time = time.time()

    # Setup multi-processing.
    num_workers = 40
    fin = open(input_file, 'r', encoding='utf-8')
    pool = multiprocessing.Pool(num_workers)
    process_doc_partial = partial(process_doc, args=args)
    processed_docs = pool.imap(process_doc_partial, fin, 500)

    # Process documents.
    for output, text, document, to_filter in processed_docs:
        num_docs += 1

        num_remove_512 += 1 if output['remove_512'] else 0
        num_remove_java += 1 if output['remove_256_javascript'] else 0
        num_remove_512_non_english += 1 if output['remove_512_non_english'] \
            else 0
        num_ftfy_fix_text += 1 if output['ftfy_fix_text'] else 0
        num_general_cleaning += 1 if output['general_cleaning'] else 0

        document['text'] = text
        myjson = json.dumps(document, ensure_ascii=False)

        if to_filter:
            output_filtered.write(myjson.encode('utf-8'))
            output_filtered.write('\n'.encode('utf-8'))
        else:
            output_cleaned.write(myjson.encode('utf-8'))
            output_cleaned.write('\n'.encode('utf-8'))

        if num_docs % args.log_interval == 0:
            print('    processed {:9d} documents in {:.2f} seconds ...'.format(num_docs,
                                                                               time.time() - start_time),
                  flush=True)

    # Close the file.
    output_cleaned.close()
    output_filtered.close()
    fin.close()

    # Print stats.
    print('  >> total docs: {} remove_512 {} remove_256_javascript {} '\
        'remove_512_non_english {} ftfy_fix_text {} general_cleaning {}'.\
        format(num_docs, num_remove_512, num_remove_java,\
        num_remove_512_non_english, num_ftfy_fix_text, \
        num_general_cleaning), flush=True)


if __name__ == '__main__':

    print('parsing the arguments ...')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', nargs = '*', required=True, default=\
                        None, help = 'Input json files that needs to be'\
                        ' cleaned')
    parser.add_argument('--tasks', nargs = '*', required=True, default=None,\
                        help = 'Tasks to perform on the input files, ' \
                        'such as remove_512, remove_256_javascript, ' \
                        'remove_512_non_english, ftfy_fix_text, and ' \
                        'general_cleaning. 256 or 512 means the number' \
                        ' of characters.')

    parser.add_argument('--output-path', type=str, default=None, help='Directory where the output should go')
    parser.add_argument('--log-interval', type=int, default=100, help='Log interval')

    args = parser.parse_args()

    print('cleanup dataset ...')

    for input_file in args.input_files:
        input_filename, input_filename_ext = os.path.splitext(Path(input_file)\
            .name)

        output_f_cleaned = os.path.join(args.output_path, input_filename + \
            "_cleaned" + input_filename_ext)
        output_f_filtered = os.path.join(args.output_path, input_filename + \
            "_filtered" + input_filename_ext)

        process_set(args, input_file, output_f_cleaned, output_f_filtered)

    print('done :-)', flush=True)
