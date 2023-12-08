from colossalai.inference.core.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine


def test_inference_engine():
    inference_config = InferenceConfig("/llama")
    inference_engine = InferenceEngine(inference_config)

    inputs = [
        "介绍一下北京",
        "介绍一下武汉",
    ]

    outputs = inference_engine.generate(inputs)

    for s1, s2 in zip(inputs, outputs):
        assert s1 == s2


if __name__ == "__main__":
    test_inference_engine()
