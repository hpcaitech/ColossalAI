/*
 coding=utf-8
 Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;


void build_blending_indices(py::array_t<uint8_t>& dataset_index,
			    py::array_t<int64_t>& dataset_sample_index,
			    const py::array_t<double>& weights,
			    const int32_t num_datasets,
			    const int64_t size, const bool verbose) {
  /* Given multiple datasets and a weighting array, build samples
   such that it follows those wieghts.*/

  if (verbose) {
    std::cout << "> building indices for blendable datasets ..." << std::endl;
  }

  // Get the pointer access without the checks.
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
  auto weights_ptr = weights.unchecked<1>();

  // Initialize buffer for number of samples used for each dataset.
  int64_t current_samples[num_datasets];
  for(int64_t i = 0; i < num_datasets; ++i) {
    current_samples[i] = 0;
  }

  // For each sample:
  for(int64_t sample_idx = 0; sample_idx < size; ++sample_idx) {

    // Determine where the max error in sampling is happening.
    auto sample_idx_double = std::max(static_cast<double>(sample_idx), 1.0);
    int64_t max_error_index = 0;
    double max_error = weights_ptr[0] * sample_idx_double -
      static_cast<double>(current_samples[0]);
    for (int64_t dataset_idx = 1; dataset_idx < num_datasets; ++dataset_idx) {
      double error = weights_ptr[dataset_idx] * sample_idx_double -
	static_cast<double>(current_samples[dataset_idx]);
      if (error > max_error) {
	max_error = error;
	max_error_index = dataset_idx;
      }
    }

    // Populate the indices.
    dataset_index_ptr[sample_idx] = static_cast<uint8_t>(max_error_index);
    dataset_sample_index_ptr[sample_idx] = current_samples[max_error_index];

    // Update the total samples.
    current_samples[max_error_index] += 1;
    
  }

  // print info
  if (verbose) {
    std::cout << " > sample ratios:" << std::endl;
    for (int64_t dataset_idx = 0; dataset_idx < num_datasets; ++dataset_idx) {
      auto ratio = static_cast<double>(current_samples[dataset_idx]) /
	static_cast<double>(size);
      std::cout << "   dataset " << dataset_idx << ", input: " <<
	weights_ptr[dataset_idx] << ", achieved: " << ratio << std::endl; 
    }
  }

}


py::array build_sample_idx(const py::array_t<int32_t>& sizes_,
			   const py::array_t<int32_t>& doc_idx_,
			   const int32_t seq_length,
			   const int32_t num_epochs,
			   const int64_t tokens_per_epoch) {
    /* Sample index (sample_idx) is used for gpt2 like dataset for which
       the documents are flattened and the samples are built based on this
       1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
       where [..., 0] contains the index into `doc_idx` and [..., 1] is the
       starting offset in that document.*/

    // Consistency checks.
    assert(seq_length > 1);
    assert(num_epochs > 0);
    assert(tokens_per_epoch > 1);

    // Remove bound checks.
    auto sizes = sizes_.unchecked<1>();
    auto doc_idx = doc_idx_.unchecked<1>();

    // Mapping and it's length (1D).
    int64_t num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length;
    int32_t* sample_idx = new int32_t[2*(num_samples+1)];

    cout << "    using:" << endl << std::flush;
    cout << "     number of documents:       " <<
      doc_idx_.shape(0) / num_epochs << endl << std::flush;
    cout << "     number of epochs:          " << num_epochs <<
      endl << std::flush;
    cout << "     sequence length:           " << seq_length <<
      endl << std::flush;
    cout << "     total number of samples:   " << num_samples <<
      endl << std::flush;

    // Index into sample_idx.
    int64_t sample_index = 0;
    // Index into doc_idx.
    int64_t doc_idx_index = 0;
    // Begining offset for each document.
    int32_t doc_offset = 0;
    // Start with first document and no offset.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;

    while (sample_index <= num_samples) {
        // Start with a fresh sequence.
      int32_t remaining_seq_length = seq_length + 1;
      while (remaining_seq_length != 0) {
            // Get the document length.
	auto doc_id = doc_idx[doc_idx_index];
	auto doc_length = sizes[doc_id] - doc_offset;
	// And add it to the current sequence.
	remaining_seq_length -= doc_length;
	// If we have more than a full sequence, adjust offset and set
	// remaining length to zero so we return from the while loop.
	// Note that -1 here is for the same reason we have -1 in
	// `_num_epochs` calculations.
	if (remaining_seq_length <= 0) {
	  doc_offset += (remaining_seq_length + doc_length - 1);
	  remaining_seq_length = 0;
	} else {
	  // Otherwise, start from the begining of the next document.
	  ++doc_idx_index;
	  doc_offset = 0;
	}
      }
      // Record the sequence.
      sample_idx[2 * sample_index] = doc_idx_index;
      sample_idx[2 * sample_index + 1] = doc_offset;
      ++sample_index;
    }

    // Method to deallocate memory.
    py::capsule free_when_done(sample_idx, [](void *mem_) {
	int32_t *mem = reinterpret_cast<int32_t*>(mem_);
	delete[] mem;
      });

    // Return the numpy array.
    const auto byte_size = sizeof(int32_t);
    return py::array(std::vector<int64_t>{num_samples+1, 2}, // shape
                     {2*byte_size, byte_size}, // C-style contiguous strides
                     sample_idx, // the data pointer
                     free_when_done); // numpy array references
    
}


inline int32_t get_target_sample_len(const int32_t short_seq_ratio,
				     const int32_t max_length,
				     std::mt19937& rand32_gen) {
    /* Training sample length. */
    if (short_seq_ratio == 0) {
      return max_length;
    }
    const auto random_number = rand32_gen();
    if ((random_number % short_seq_ratio) == 0) {
      return 2 + random_number % (max_length - 1);
    }
    return max_length;
}


template<typename DocIdx>
py::array build_mapping_impl(const py::array_t<int64_t>& docs_,
                             const py::array_t<int32_t>& sizes_,
                             const int32_t num_epochs,
                             const uint64_t max_num_samples,
                             const int32_t max_seq_length,
                             const double short_seq_prob,
                             const int32_t seed,
			     const bool verbose,
			     const int32_t min_num_sent) {
    /* Build a mapping of (start-index, end-index, sequence-length) where
       start and end index are the indices of the sentences in the sample
       and sequence-length is the target sequence length.
    */

    // Consistency checks.
    assert(num_epochs > 0);
    assert(max_seq_length > 1);
    assert(short_seq_prob >= 0.0);
    assert(short_seq_prob <= 1.0);
    assert(seed > 0);

    // Remove bound checks.
    auto docs = docs_.unchecked<1>();
    auto sizes = sizes_.unchecked<1>();

    // For efficiency, convert probability to ratio. Note: rand() generates int.
    int32_t short_seq_ratio = 0;
    if (short_seq_prob > 0) {
      short_seq_ratio = static_cast<int32_t>(round(1.0 / short_seq_prob));
    }

    if (verbose) {
        const auto sent_start_index = docs[0];
	const auto sent_end_index = docs[docs_.shape(0) - 1];
	const auto num_sentences = sent_end_index - sent_start_index;
	cout << "    using:" << endl << std::flush;
	cout << "     number of documents:            " << docs_.shape(0) - 1 <<
	  endl << std::flush;
	cout << "     sentences range:                [" << sent_start_index <<
	", " << sent_end_index << ")" << endl << std::flush;
	cout << "     total number of sentences:      " << num_sentences <<
	  endl << std::flush;
	cout << "     number of epochs:               " << num_epochs <<
	  endl << std::flush;
	cout << "     maximum number of samples:      " << max_num_samples <<
	  endl << std::flush;
	cout << "     maximum sequence length:        " << max_seq_length <<
	  endl << std::flush;
	cout << "     short sequence probability:     " << short_seq_prob <<
	endl << std::flush;
	cout << "     short sequence ration (1/prob): " << short_seq_ratio <<
	  endl << std::flush;
	cout << "     seed:                           " << seed << endl <<
	  std::flush;
    }

    // Mapping and it's length (1D).
    int64_t num_samples = -1;
    DocIdx* maps = NULL;

    // Perform two iterations, in the first iteration get the size
    // and allocate memory and in the second iteration populate the map.
    bool second = false;
    for (int32_t iteration=0; iteration<2; ++iteration) {

        // Set the seed so both iterations produce the same results.
        std::mt19937 rand32_gen(seed);

        // Set the flag on second iteration.
        second = (iteration == 1);

        // Counters:
        uint64_t empty_docs = 0;
        uint64_t one_sent_docs = 0;
	uint64_t long_sent_docs = 0;

        // Current map index.
        uint64_t map_index = 0;

        // For each epoch:
        for (int32_t epoch=0; epoch<num_epochs; ++epoch) {
            if (map_index >= max_num_samples) {
	        if (verbose && (!second)) {
		  cout << "    reached " << max_num_samples << " samples after "
		       << epoch << " epochs ..." << endl << std::flush;
		}
                break;
            }
            // For each document:
            for (int32_t doc=0; doc<(docs.shape(0) - 1); ++doc) {

                // Document sentences are in [sent_index_first, sent_index_last)
                const auto sent_index_first = docs[doc];
                const auto sent_index_last = docs[doc + 1];

                // At the begining of the document previous index is the
		// start index.
                auto prev_start_index = sent_index_first;

                // Remaining documents.
                auto num_remain_sent = sent_index_last - sent_index_first;

                // Some bookkeeping
                if ((epoch == 0) && (!second)) {
                    if (num_remain_sent == 0) {
		        ++empty_docs;
                    }
                    if (num_remain_sent == 1) {
		        ++one_sent_docs;
                    }
                }

		// Detect documents with long sentences.
		bool contains_long_sentence = false;
		if (num_remain_sent > 1) {
		    for (auto sent_index=sent_index_first;
			 sent_index < sent_index_last; ++sent_index) {
		        if (sizes[sent_index] > LONG_SENTENCE_LEN){
			    if ((epoch == 0) && (!second)) {
			        ++long_sent_docs;
			    }
			    contains_long_sentence = true;
			    break;
			}
		    }
		}

                // If we have more than two sentences.
                if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence)) {

                    // Set values.
                    auto seq_len = int32_t{0};
                    auto num_sent = int32_t{0};
                    auto target_seq_len = get_target_sample_len(short_seq_ratio,
								max_seq_length,
								rand32_gen);

                    // Loop through sentences.
                    for (auto sent_index=sent_index_first;
                         sent_index < sent_index_last; ++sent_index) {

		        // Add the size and number of sentences.
		        seq_len += sizes[sent_index];
		        ++num_sent;
			--num_remain_sent;

			// If we have reached the target length.
			// and if not only one sentence is left in the document.
			// and if we have at least two sentneces.
			// and if we have reached end of the document.
			if (((seq_len >= target_seq_len) &&
			     (num_remain_sent > 1) &&
			     (num_sent >= min_num_sent) ) || (num_remain_sent == 0)) {

			    // Check for overflow.
			    if ((3 * map_index + 2) >
				std::numeric_limits<int64_t>::max()) {
			        cout << "number of samples exceeded maximum "
				     << "allowed by type int64: "
				     << std::numeric_limits<int64_t>::max()
				     << endl;
				throw std::overflow_error("Number of samples");
			    }

			    // Populate the map.
			    if (second) {
			        const auto map_index_0 = 3 * map_index;
				maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
				maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
				maps[map_index_0 + 2] = static_cast<DocIdx>(target_seq_len);
			    }

			    // Update indices / counters.
			    ++map_index;
			    prev_start_index = sent_index + 1;
			    target_seq_len = get_target_sample_len(short_seq_ratio,
								   max_seq_length,
								   rand32_gen);
			    seq_len = 0;
			    num_sent = 0;
			}

                    } // for (auto sent_index=sent_index_first; ...
                } // if (num_remain_sent > 1) {
            } // for (int doc=0; doc < num_docs; ++doc) {
        } // for (int epoch=0; epoch < num_epochs; ++epoch) {

        if (!second) {
	    if (verbose) {
	        cout << "   number of empty documents: " << empty_docs <<
		  endl << std::flush;
		cout << "   number of documents with one sentence: " <<
		  one_sent_docs << endl << std::flush;
		cout << "   number of documents with long sentences: " <<
		  long_sent_docs << endl << std::flush;
		cout << "   will create mapping for " << map_index <<
		  " samples" << endl << std::flush;
	    }
	    assert(maps == NULL);
	    assert(num_samples < 0);
            maps = new DocIdx[3*map_index];
            num_samples = static_cast<int64_t>(map_index);
        }

    } // for (int iteration=0; iteration < 2; ++iteration) {

    // Shuffle.
    // We need a 64 bit random number generator as we might have more
    // than 2 billion samples.
    std::mt19937_64 rand64_gen(seed + 1);
    for (auto i=(num_samples - 1); i > 0; --i) {
      const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
      const auto i0 = 3 * i;
      const auto j0 = 3 * j;
      // Swap values.
      swap(maps[i0], maps[j0]);
      swap(maps[i0 + 1], maps[j0 + 1]);
      swap(maps[i0 + 2], maps[j0 + 2]);
    }

    // Method to deallocate memory.
    py::capsule free_when_done(maps, [](void *mem_) {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem;
        });

    // Return the numpy array.
    const auto byte_size = sizeof(DocIdx);
    return py::array(std::vector<int64_t>{num_samples, 3}, // shape
                     {3*byte_size, byte_size}, // C-style contiguous strides
                     maps, // the data pointer
                     free_when_done); // numpy array references

}


py::array build_mapping(const py::array_t<int64_t>& docs_,
                        const py::array_t<int>& sizes_,
                        const int num_epochs,
                        const uint64_t max_num_samples,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const int seed,
			const bool verbose,
			const int32_t min_num_sent) {

    if (sizes_.size() > std::numeric_limits<uint32_t>::max()) {
        if (verbose) {
	   cout << "    using uint64 for data mapping..." << endl << std::flush;
	}
	return build_mapping_impl<uint64_t>(docs_, sizes_, num_epochs,
					    max_num_samples, max_seq_length,
					    short_seq_prob, seed, verbose,
					    min_num_sent);
    } else {
       if (verbose) {
	   cout << "    using uint32 for data mapping..." << endl << std::flush;
       }
       return build_mapping_impl<uint32_t>(docs_, sizes_, num_epochs,
					   max_num_samples, max_seq_length,
					   short_seq_prob, seed, verbose,
					   min_num_sent);
    }
}

template<typename DocIdx>
py::array build_blocks_mapping_impl(const py::array_t<int64_t>& docs_,
                                    const py::array_t<int32_t>& sizes_,
                                    const py::array_t<int32_t>& titles_sizes_,
                                    const int32_t num_epochs,
                                    const uint64_t max_num_samples,
                                    const int32_t max_seq_length,
                                    const int32_t seed,
                                    const bool verbose,
                                    const bool use_one_sent_blocks) {
    /* Build a mapping of (start-index, end-index, sequence-length) where
       start and end index are the indices of the sentences in the sample
       and sequence-length is the target sequence length.
    */

    // Consistency checks.
    assert(num_epochs > 0);
    assert(max_seq_length > 1);
    assert(seed > 0);

    // Remove bound checks.
    auto docs = docs_.unchecked<1>();
    auto sizes = sizes_.unchecked<1>();
    auto titles_sizes = titles_sizes_.unchecked<1>();

    if (verbose) {
        const auto sent_start_index = docs[0];
        const auto sent_end_index = docs[docs_.shape(0) - 1];
        const auto num_sentences = sent_end_index - sent_start_index;
        cout << "    using:" << endl << std::flush;
        cout << "     number of documents:            " << docs_.shape(0) - 1 <<
          endl << std::flush;
        cout << "     sentences range:                [" << sent_start_index <<
        ", " << sent_end_index << ")" << endl << std::flush;
        cout << "     total number of sentences:      " << num_sentences <<
          endl << std::flush;
        cout << "     number of epochs:               " << num_epochs <<
          endl << std::flush;
        cout << "     maximum number of samples:      " << max_num_samples <<
          endl << std::flush;
        cout << "     maximum sequence length:        " << max_seq_length <<
          endl << std::flush;
        cout << "     seed:                           " << seed << endl <<
          std::flush;
    }

    // Mapping and its length (1D).
    int64_t num_samples = -1;
    DocIdx* maps = NULL;

    // Acceptable number of sentences per block.
    int min_num_sent = 2;
    if (use_one_sent_blocks) {
        min_num_sent = 1;
    }

    // Perform two iterations, in the first iteration get the size
    // and allocate memory and in the second iteration populate the map.
    bool second = false;
    for (int32_t iteration=0; iteration<2; ++iteration) {

        // Set the flag on second iteration.
        second = (iteration == 1);

        // Current map index.
        uint64_t map_index = 0;

        uint64_t empty_docs = 0;
        uint64_t one_sent_docs = 0;
        uint64_t long_sent_docs = 0;
        // For each epoch:
        for (int32_t epoch=0; epoch<num_epochs; ++epoch) {
            // assign every block a unique id
            int32_t block_id = 0;

            if (map_index >= max_num_samples) {
                if (verbose && (!second)) {
                cout << "    reached " << max_num_samples << " samples after "
                     << epoch << " epochs ..." << endl << std::flush;
                }
                break;
            }
            // For each document:
            for (int32_t doc=0; doc<(docs.shape(0) - 1); ++doc) {

                // Document sentences are in [sent_index_first, sent_index_last)
                const auto sent_index_first = docs[doc];
                const auto sent_index_last = docs[doc + 1];
                const auto target_seq_len = max_seq_length - titles_sizes[doc];

                // At the begining of the document previous index is the
                // start index.
                auto prev_start_index = sent_index_first;

                // Remaining documents.
                auto num_remain_sent = sent_index_last - sent_index_first;

                // Some bookkeeping
                if ((epoch == 0) && (!second)) {
                    if (num_remain_sent == 0) {
		                ++empty_docs;
                    }
                    if (num_remain_sent == 1) {
		                ++one_sent_docs;
                    }
                }
                // Detect documents with long sentences.
                bool contains_long_sentence = false;
                if (num_remain_sent >= min_num_sent) {
                    for (auto sent_index=sent_index_first;
                    sent_index < sent_index_last; ++sent_index) {
                        if (sizes[sent_index] > LONG_SENTENCE_LEN){
                            if ((epoch == 0) && (!second)) {
                                ++long_sent_docs;
                            }
                            contains_long_sentence = true;
                            break;
                        }
                    }
                }
                // If we have enough sentences and no long sentences.
                if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence)) {

                    // Set values.
                    auto seq_len = int32_t{0};
                    auto num_sent = int32_t{0};

                    // Loop through sentences.
                    for (auto sent_index=sent_index_first;
                         sent_index < sent_index_last; ++sent_index) {

                            // Add the size and number of sentences.
                            seq_len += sizes[sent_index];
                            ++num_sent;
                            --num_remain_sent;

                        // If we have reached the target length.
                        // and there are an acceptable number of sentences left
                        // and if we have at least the minimum number of sentences.
                        // or if we have reached end of the document.
                        if (((seq_len >= target_seq_len) &&
                             (num_remain_sent >= min_num_sent) &&
                             (num_sent >= min_num_sent) ) || (num_remain_sent == 0)) {

                            // Populate the map.
                            if (second) {
                                const auto map_index_0 = 4 * map_index;
                                // Each sample has 4 items: the starting sentence index, ending sentence index,
                                // the index of the document from which the block comes (used for fetching titles)
                                // and the unique id of the block (used for creating block indexes)

                                maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
                                maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
                                maps[map_index_0 + 2] = static_cast<DocIdx>(doc);
                                maps[map_index_0 + 3] = static_cast<DocIdx>(block_id);
                            }

                            // Update indices / counters.
                            ++map_index;
                            ++block_id;
                            prev_start_index = sent_index + 1;
                            seq_len = 0;
                            num_sent = 0;
                        }
                    } // for (auto sent_index=sent_index_first; ...
                } // if (num_remain_sent > 1) {
            } // for (int doc=0; doc < num_docs; ++doc) {
        } // for (int epoch=0; epoch < num_epochs; ++epoch) {

        if (!second) {
            if (verbose) {
	        cout << "   number of empty documents: " << empty_docs <<
              endl << std::flush;
            cout << "   number of documents with one sentence: " <<
              one_sent_docs << endl << std::flush;
            cout << "   number of documents with long sentences: " <<
              long_sent_docs << endl << std::flush;
            cout << "   will create mapping for " << map_index <<
              " samples" << endl << std::flush;
            }
            assert(maps == NULL);
            assert(num_samples < 0);
            maps = new DocIdx[4*map_index];
            num_samples = static_cast<int64_t>(map_index);
        }

    } // for (int iteration=0; iteration < 2; ++iteration) {

    // Shuffle.
    // We need a 64 bit random number generator as we might have more
    // than 2 billion samples.
    std::mt19937_64 rand64_gen(seed + 1);
    for (auto i=(num_samples - 1); i > 0; --i) {
        const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
        const auto i0 = 4 * i;
        const auto j0 = 4 * j;
        // Swap values.
        swap(maps[i0], maps[j0]);
        swap(maps[i0 + 1], maps[j0 + 1]);
        swap(maps[i0 + 2], maps[j0 + 2]);
        swap(maps[i0 + 3], maps[j0 + 3]);
    }

    // Method to deallocate memory.
    py::capsule free_when_done(maps, [](void *mem_) {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem;
        });

    // Return the numpy array.
    const auto byte_size = sizeof(DocIdx);
    return py::array(std::vector<int64_t>{num_samples, 4}, // shape
                     {4*byte_size, byte_size}, // C-style contiguous strides
                     maps, // the data pointer
                     free_when_done); // numpy array references

}

py::array build_blocks_mapping(const py::array_t<int64_t>& docs_,
                               const py::array_t<int>& sizes_,
                               const py::array_t<int>& titles_sizes_,
                               const int num_epochs,
                               const uint64_t max_num_samples,
                               const int max_seq_length,
                               const int seed,
                    const bool verbose,
                    const bool use_one_sent_blocks) {

    if (sizes_.size() > std::numeric_limits<uint32_t>::max()) {
        if (verbose) {
	   cout << "    using uint64 for data mapping..." << endl << std::flush;
	}
	return build_blocks_mapping_impl<uint64_t>(docs_, sizes_, titles_sizes_,
	                    num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
    } else {
       if (verbose) {
	   cout << "    using uint32 for data mapping..." << endl << std::flush;
       }
       return build_blocks_mapping_impl<uint32_t>(docs_, sizes_, titles_sizes_,
                        num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
    }
}

PYBIND11_MODULE(helpers, m) {
    m.def("build_mapping", &build_mapping);
    m.def("build_blocks_mapping", &build_blocks_mapping);
    m.def("build_sample_idx", &build_sample_idx);
    m.def("build_blending_indices", &build_blending_indices);
}
