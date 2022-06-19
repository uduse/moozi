import numpy as np
from moozi import ScalarTransform


def test_scalar_transform():
    scalar_transform = ScalarTransform(support_min=-10, support_max=10)

    inputs = np.random.randn(5) ** 100
    transformed = scalar_transform.transform(inputs)
    outputs = scalar_transform.inverse_transform(transformed)
    assert np.allclose(inputs, outputs, atol=1e-3)

# def test_mask_tape()
# # %%
# tape = {0: "a", 1: "b", 2: "c"}
# print(tape)
# with mask_tape(tape, {0}) as masked:
#     print(masked)
#     masked[1] = "bbb"
#     print(masked)
# print(tape)
