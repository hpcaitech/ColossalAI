from typing import Dict, List
from dataclasses import dataclass

# This file includes data structure used by Pipeline Middleware.

@dataclass
class ValPosition:
    partition_id: int
    offset: int
    
    def __str__(self) -> str:
        res = f'[partition_id:{self.partition_id},offset:{self.offset}]'
        return res
    
    def __repr__(self) -> str:
        return self.__str__()

class PartitionInputVal(object):
    def __init__(self, partition_id, offset) -> None:
        # every input from which partition_id and which offset
        val_pos = ValPosition(partition_id, offset)
        self._from_partition_and_offset: ValPosition = val_pos
        
    def get(self):
        return self._from_partition_and_offset
    
    def __str__(self) -> str:
        res = ''
        res += f'<-({self._from_partition_and_offset})'
        return res
    
    def __repr__(self) -> str:
        return self.__str__()
    
class PartitionOutputVal(object):
    def __init__(self) -> None:
        # every output to which partition_id and which offset
        self._to_partition_and_offset: List[ValPosition] = []
        
    def add(self, partition_id, offset):
        val_pos = ValPosition(partition_id, offset)
        self._to_partition_and_offset.append(val_pos)
        
    def get(self):
        return self._to_partition_and_offset
        
    def __str__(self) -> str:
        res = ''
        res += '->('
        for val_pos in self._to_partition_and_offset:
            res += f'{val_pos},'
        res += ')'
        return res
    
    def __repr__(self) -> str:
        return self.__str__()

class Partition(object):
    def __init__(self) -> None:
        self._input_vals: List[PartitionInputVal] = []
        self._output_vals: List[PartitionOutputVal] = []
        
    def add_input_val(self, input_val: PartitionInputVal):
        self._input_vals.append(input_val)
        
    def add_output_val(self, output_val: PartitionOutputVal):
        self._output_vals.append(output_val)
        
    def get_input_vals(self):
        return self._input_vals
    
    def get_output_vals(self):
        return self._output_vals
    
    # get the output offsets sent to dst_partition_id
    def get_output_offsets(self, dst_partition_id):
        res = []
        for offset, output_val in enumerate(self._output_vals):
            outputs = output_val.get()
            for val_pos in outputs:
                if val_pos.partition_id == dst_partition_id:
                    res.append(offset)
            
        return res
    
    # get all input dst partition_ids
    def get_input_partition_ids(self):
        res = []
        for input_val in self._input_vals:
            val_pos = input_val.get()
            if val_pos.partition_id not in res:
                res.append(val_pos.partition_id)
        return res
    
    # get all output dst partition_ids
    def get_output_partition_ids(self):
        res = []
        for output_val in self._output_vals:
            outputs = output_val.get()
            for val_pos in outputs:
                if val_pos.partition_id not in res:
                    res.append(val_pos.partition_id)
        return res
        
    def __str__(self) -> str:
        res = ''
        res += f'  input:\n'
        res += f'    length:{len(self._input_vals)}\n'
        for i, input_val in enumerate(self._input_vals):
            res += f'    offset={i}:{input_val}\n'
            
        res += f'  output:\n'
        res += f'    length:{len(self._output_vals)}\n'
        for i, output_val in enumerate(self._output_vals):
            res += f'    offset={i}:{output_val}\n'
        
        return res
    
    def __repr__(self) -> str:
        return self.__str__()

# This class is a middleware between partition splitter
# and Pipeline Scheduler. It records the graph info about
# partition input/output and provides it to scheduler.
# There are three kinds of partition in Pipeline Middleware Design
# which represents the whole process of a model execution: input-fwd-output
# 1. input_partition: records the input of a model.
# 2. mid_partition: record the splitted forwards execution of a model.
# 3. output_partition: records the output of a model.
# attributes:
#   _partitions: include all partitions
#   _input_partition_id: the key represents input_partition
#   _output_partition_id: the key represents output_partition
class Topo(object):
    def __init__(self, input_partition_id=None, output_partition_id=None) -> None:
        self._partitions: Dict[int, Partition] = {}
        self._input_partition_id = input_partition_id
        self._output_partition_id = output_partition_id
        
    def set_input_partition_id(self, partition_id: int):
        self._input_partition_id = partition_id
    
    def set_output_partition_id(self, partition_id: int):
        self._output_partition_id = partition_id
        
    def get_input_partition_id(self):
        return self._input_partition_id
    
    def get_output_partition_id(self):
        return self._output_partition_id
    
    def set_partitions(self, partition_id: int, partition: Partition):
        self._partitions[partition_id] = partition
        
    def get_mid_partitions(self):
        res = {} #{partition_id: Partition}
        for partition_id, partition in self._partitions.items():
            if self._input_partition_id == partition_id or self._output_partition_id == partition_id:
                continue
            res[partition_id] = partition
        return res
    
    def get_mid_partition_ids(self):
        return list(self.get_mid_partitions().keys())
    
    def get_input_partition(self):
        if self._input_partition_id is not None:
            return self._partitions[self._input_partition_id]
        return None
    
    def get_output_partition(self):
        if self._output_partition_id is not None:
            return self._partitions[self._output_partition_id]
        return None

    def get_partition_by_id(self, partition_id):
        return self._partitions[partition_id]
        
    def __str__(self) -> str:
        res = ''
        if len(self._partitions) == 0:
            return 'Empty Topo Graph.'

        input_part = self.get_input_partition()
        if input_part is not None:
            res += '{\n'
            res += f'InputPartition:\n  partition_id={self._input_partition_id}\n{input_part}'
            res += '}\n'
        
        mid_parts = self.get_mid_partitions()
        for i, (partition_id, part) in enumerate(mid_parts.items()):
            res += '{\n'
            res += f'SubPartition_{i}:\n  partition_id={partition_id}\n  {part}'
            res += '}\n'
            
        output_part = self.get_output_partition()
        if output_part is not None:
            res += '{\n'
            res += f'OutputPartition:\n  partition_id={self._output_partition_id}\n{output_part}'
            res += '}\n'
            
        return res
    
    def __repr__(self) -> str:
        return self.__str__()
        