from colossalai.elixir.tracer.param_tracer import generate_tf_order
from tests.test_elixir.utils import TEST_MODELS


def test_tf_forward_backward():
    model_fn, data_fn = TEST_MODELS.get('gpt2_micro')
    model = model_fn()
    data = data_fn()

    def forward_backward_fn(local_model, local_input):
        local_model(**local_input).backward()

    # model.gradient_checkpointing_enable()
    tf_order = generate_tf_order(model, data, forward_backward_fn)
    params_per_step = tf_order['params_per_step']
    assert len(params_per_step) == 32

    model.gradient_checkpointing_enable()
    tf_order = generate_tf_order(model, data, forward_backward_fn)
    params_per_step = tf_order['params_per_step']
    checkpoint_info = tf_order['checkpoint_info']
    for i, step in enumerate(params_per_step):
        print(f'step {i}: {step}')
    for c in checkpoint_info:
        print(f'checkpoint info: {c}')
    assert len(params_per_step) == 44

    assert data['input_ids'].device.type == 'cpu'
    assert data['attention_mask'].device.type == 'cpu'
    for param in model.parameters():
        assert param.device.type == 'cpu'


if __name__ == '__main__':
    test_tf_forward_backward()
