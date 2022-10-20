from .nn import (
    NNArchitecture,
    NNSpec,
    NNOutput,
    NNModel,
    RootFeatures,
    TransitionFeatures,
    make_model,
)
from .naive import NaiveArchitecture
from .mlp import MLPArchitecture, MLPSpec
from .resnet import ResNetArchitecture, ResNetSpec
from .resnet_v2 import ResNetV2Architecture, ResNetV2Spec

import importlib


def get(name):
    return getattr(importlib.import_module("moozi.nn"), name)
