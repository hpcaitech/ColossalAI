import time
from pathlib import Path

import onnx
import tensorrt as trt
from onnx import TensorProto, helper
from tensorrt_llm.logger import logger
from tensorrt_llm.network import Network


def to_onnx(network: Network, path: Path) -> None:
    inputs = []
    for i in range(network.num_inputs):
        network_input = network.get_input(i)
        inputs.append(
            helper.make_tensor_value_info(
                network_input.name, trt_dtype_to_onnx(network_input.dtype), list(network_input.shape)
            )
        )

    outputs = []
    for i in range(network.num_outputs):
        network_output = network.get_output(i)
        outputs.append(
            helper.make_tensor_value_info(
                network_output.name, trt_dtype_to_onnx(network_output.dtype), list(network_output.shape)
            )
        )

    nodes = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_inputs = []
        for j in range(layer.num_inputs):
            ipt = layer.get_input(j)
            if ipt is not None:
                layer_inputs.append(layer.get_input(j).name)
        layer_outputs = [layer.get_output(j).name for j in range(layer.num_outputs)]
        nodes.append(
            helper.make_node(
                str(layer.type), name=layer.name, inputs=layer_inputs, outputs=layer_outputs, domain="com.nvidia"
            )
        )

    onnx_model = helper.make_model(
        helper.make_graph(nodes, "attention", inputs, outputs, initializer=None), producer_name="NVIDIA"
    )
    onnx.save(onnx_model, path)


def trt_dtype_to_onnx(dtype) -> None:
    if dtype == trt.float16:
        return TensorProto.DataType.FLOAT16
    elif dtype == trt.float32:
        return TensorProto.DataType.FLOAT
    elif dtype == trt.int32:
        return TensorProto.DataType.INT32
    else:
        raise TypeError("%s is not supported" % dtype)


def get_engine_name(model: str, dtype: str, tp_size: int, pp_size: int, rank: int) -> str:
    if pp_size == 1:
        return "{}_{}_tp{}_rank{}.engine".format(model, dtype, tp_size, rank)
    return "{}_{}_tp{}_pp{}_rank{}.engine".format(model, dtype, tp_size, pp_size, rank)


def serialize_engine(engine: trt.IHostMemory, path: Path) -> None:
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    with open(path, "wb") as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


def process_output(output_ids, input_lengths, max_output_len, tokenizer, output_csv, output_npy) -> str:
    num_beams = output_ids.size(1)
    outputs_text = []
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][: input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs, skip_special_tokens=True)
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                outputs_text.append(input_text + "\n" + output_text)

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype="int32")
        np.save(output_file, outputs)

    return outputs_text


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out
