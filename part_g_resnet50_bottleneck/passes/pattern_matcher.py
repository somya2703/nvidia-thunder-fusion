# Simple utilities to detect Conv->BN->ReLU patterns using torch.fx
from torch.fx import symbolic_trace
import torch
import os

def find_conv_bn_relu_patterns(model: torch.nn.Module):
    gm = symbolic_trace(model)
    patterns = []
    for node in gm.graph.nodes:
        # crude detection: look for call_function F.relu whose arg is a call to add
        if node.op == 'call_function' and node.target == torch.nn.functional.relu:
            arg = node.args[0]
            # check arg is add of conv output + bias OR direct call 'call_function' add
            if getattr(arg, 'op', None) == 'call_function' and arg.target == torch.ops.aten.add.Tensor:
                conv_node = arg.args[0]
                if getattr(conv_node, 'op', None) == 'call_function' and conv_node.target == torch.ops.aten.convolution.default:
                    patterns.append((conv_node, arg, node))
    return patterns