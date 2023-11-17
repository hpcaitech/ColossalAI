from pathlib import Path
from typing import Union, Tuple
import tensorrt_llm
import torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from utils import process_output, throttle_generator

class EngineRunnerBase:
    
    def __init__(self):
        self._decoder: tensorrt_llm.runtime.GenerationSession = None
    
    def read_config(self, config_path: Path) -> Tuple[ModelConfig, int, int, str]:
        pass
    
    def _parse_input(self, 
                    input_text: str, 
                    input_file: str, 
                    tokenizer, end_id: int,
                    remove_input_padding: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def _generate_output(self,
                        input_ids: torch.Tensor,
                        context_lengths: torch.Tensor,
                        sampling_config: SamplingConfig,
                        prompt_embedding_table: torch.Tensor = None,
                        tasks: torch.Tensor = None,
                        prompt_vocab_size: torch.Tensor = None,
                        stop_words_list=None,
                        bad_words_list=None,
                        no_repeat_ngram_size=None,
                        streaming: bool = False,
                        output_sequence_lengths: bool = False,
                        return_dict: bool = False,
                        encoder_output: torch.Tensor = None,
                        encoder_input_lengths: torch.Tensor = None
                        ) -> str
        if self._decoder == None:
            raise ValueError(f"TensorRT decode is not initialized.")
        output_gen_ids = self._decoder.decode(input_ids=input_ids,
                                              context_lengths=context_lengths,
                                              sampling_config=sampling_config,
                                              prompt_embedding_table=prompt_embedding_table,
                                              tasks=tasks,
                                              prompt_vocab_size=prompt_vocab_size,
                                              stop_words_list=stop_words_list,
                                              bad_words_list=bad_words_list,
                                              no_repeat_ngram_size=no_repeat_ngram_size,
                                              streaming=streaming,
                                              output_sequence_lengths=output_sequence_lengths,
                                              return_dict=return_dict,
                                              encoder_output=encoder_output,
                                              encoder_input_lengths=encoder_input_lengths,
                                              )
        torch.cuda.synchronize()
        if streaming:
            for output_ids in throttle_generator(output_gen_ids,
                                                 streaming_interval):
                if runtime_rank == 0:
                    process_output(output_ids, input_lengths, max_output_len,
                                   tokenizer, output_csv, output_npy)
        else:
            output_ids = output_gen_ids
            if runtime_rank == 0:
                process_output(output_ids, input_lengths, max_output_len, tokenizer,
                               output_csv, output_npy)
    