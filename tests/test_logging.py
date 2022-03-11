import numpy as np
from moozi.logging import LogDatum, LogHistogram, LogText, LogScalar
import jax.numpy as jnp


def test_data_from_dict():
    d = dict(l=[0], v=1.0, v2=2, s="str")
    assert LogDatum.from_dict(d) == [
        LogText("l", "[0]"),
        LogScalar("v", 1.0),
        LogScalar("v2", 2),
        LogText("s", "str"),
    ]
    
    d = dict(name=jnp.ones((2, 3)))
    dd = LogDatum.from_dict(d)
    assert isinstance(dd[0], LogHistogram) 
    np.testing.assert_allclose(np.array(d['name']), np.array(dd[0].values))