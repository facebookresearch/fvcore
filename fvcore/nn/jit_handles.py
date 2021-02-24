# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pyre-ignore-all-errors[2,3,33]
# NOTE: most Any type in this file should be torch._C.Value - which was not yet annotated.

import typing
from collections import Counter, OrderedDict
from typing import Any, Callable, List

from numpy import prod


Handle = Callable[[List[Any], List[Any]], typing.Counter[str]]


def generic_activation_jit(
    op_name: str,
) -> Callable[[List[Any], List[Any]], typing.Counter[str]]:
    """
    This method return a handle that counts the number of activation from the
    output shape for the specified operation.

    Args:
        op_name (str): The name of the operation.

    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(outputs: List[Any]) -> int:
        """
        This is a generic jit handle that counts the number of activations for any
        operation given the output shape.

        Args:
            outputs (list(torch._C.Value)): The output shape in the form of a list
                of jit object.

        Returns:
            int: Total number of activations for each operation.
        """
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)
        return ac_count

    return lambda inputs, outputs: Counter({op_name: _generic_activation_jit(outputs)})


def get_shape(val: Any) -> List[int]:
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        raise ValueError()


def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for fully connected layers with torch script.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flop = batch_size * input_dim * output_dim
    flop_counter = Counter({"addmm": flop})
    return flop_counter


def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for the bmm operation.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before bmm.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after bmm.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    flop_counter = Counter({"bmm": flop})
    return flop_counter


def conv_flop_count(
    x_shape: List[int], w_shape: List[int], out_shape: List[int]
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = prod(out_shape[2:])
    kernel_size = prod(w_shape[2:])
    flop = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    flop_counter = Counter({"conv": flop})
    return flop_counter


def conv_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for convolution using torch script.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before convolution.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after convolution.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of Convolution should be a list of length 12 or 13. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # starting with #40737 it will be 12) user_enabled_tf32
    assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    return conv_flop_count(x_shape, w_shape, out_shape)


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for the einsum operation. We currently support
    two einsum operations: "nct,ncp->ntp" and "ntg,ncg->nct".

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before einsum.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after einsum.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of einsum should be a list of length 2.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) == 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    else:
        raise NotImplementedError("Unsupported einsum operation.")


def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for matmul.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before matmul.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after matmul.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    flop_counter = Counter({"matmul": flop})
    return flop_counter


def batchnorm_flop_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    """
    This method counts the flops for batch norm.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before batch norm.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after batch norm.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs[0] contains the shape of the input.
    input_shape = get_shape(inputs[0])
    assert 2 <= len(input_shape) <= 5, input_shape
    flop = prod(input_shape) * 4
    flop_counter = Counter({"batchnorm": flop})
    return flop_counter
