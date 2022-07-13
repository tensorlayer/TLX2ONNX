#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import onnx
from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node


@OpMapper(["RNN"])
class RNN():
    # suppport v1-v11
    @classmethod
    def concat_params(cls, num, weight_ih, weight_hh, bias_ih, bias_hh, bidrectional):
        ih_i, hh_i, b_i = None, None, None
        if bidrectional:
            id = num * 2
            # get i-th rnn layer's weights - ih_i input to hidden
            ih_i_forward = weight_ih[id]
            ih_i_reverse = weight_ih[id + 1]
            ih_i_forward = tlx.convert_to_numpy(ih_i_forward)
            ih_i_reverse = tlx.convert_to_numpy(ih_i_reverse)
            ih_i_forward = ih_i_forward[np.newaxis, :, :]
            ih_i_reverse = ih_i_reverse[np.newaxis, :, :]
            ih_i = np.concatenate((ih_i_forward, ih_i_reverse), axis=0)

            # get i-th rnn layer's weights - hh_i hidden to hidden
            hh_i_forward = weight_hh[id]
            hh_i_reverse = weight_hh[id + 1]
            hh_i_forward = tlx.convert_to_numpy(hh_i_forward)
            hh_i_reverse = tlx.convert_to_numpy(hh_i_reverse)
            hh_i_forward = hh_i_forward[np.newaxis, :, :]
            hh_i_reverse = hh_i_reverse[np.newaxis, :, :]
            hh_i = np.concatenate((hh_i_forward, hh_i_reverse), axis=0)

            if bias_ih is not None:
                # get i-th rnn layer's bias - ih_b input to hidden
                b_ih_forward = bias_ih[id]
                b_ih_reverse = bias_ih[id + 1]
                b_ih_forward = tlx.convert_to_numpy(b_ih_forward)
                b_ih_reverse = tlx.convert_to_numpy(b_ih_reverse)
                b_ih_forward = b_ih_forward[np.newaxis, :]
                b_ih_reverse = b_ih_reverse[np.newaxis, :]
                # get i-th rnn layer's bias - hh_b hidden to hidden
                b_hh_forward = bias_hh[id]
                b_hh_reverse = bias_hh[id + 1]
                b_hh_forward = tlx.convert_to_numpy(b_hh_forward)
                b_hh_reverse = tlx.convert_to_numpy(b_hh_reverse)
                b_hh_forward = b_hh_forward[np.newaxis, :]
                b_hh_reverse = b_hh_reverse[np.newaxis, :]

                # concat bias
                b_forward = np.concatenate((b_ih_forward, b_hh_forward), axis=-1)
                b_reverse = np.concatenate((b_ih_reverse, b_hh_reverse), axis=-1)
                b_i = np.concatenate((b_forward, b_reverse), axis=0)
        else:
            # get i-th rnn layer's weights - ih_i input to hidden
            ih_i_forward = weight_ih[num]
            ih_i_forward = tlx.convert_to_numpy(ih_i_forward)
            ih_i = ih_i_forward[np.newaxis, :, :]

            # get i-th rnn layer's weights - hh_i hidden to hidden
            hh_i_forward = weight_hh[num]
            hh_i_forward = tlx.convert_to_numpy(hh_i_forward)
            hh_i = hh_i_forward[np.newaxis, :, :]

            if bias_ih is not None:
                # get i-th rnn layer's bias - ih_b input to hidden
                b_ih_forward = bias_ih[num]
                b_ih_forward = tlx.convert_to_numpy(b_ih_forward)
                b_ih_forward = b_ih_forward[np.newaxis, :]
                # get i-th rnn layer's bias - hh_b hidden to hidden
                b_hh_forward = bias_hh[num]
                b_hh_forward = tlx.convert_to_numpy(b_hh_forward)
                b_hh_forward = b_hh_forward[np.newaxis, :]

                # concat bias
                b_i = np.concatenate((b_ih_forward, b_hh_forward), axis=-1)

        return ih_i, hh_i, b_i

    @classmethod
    def concat_states(cls, num, states, bidrectional):
        states_i = None
        states = tlx.convert_to_numpy(states)
        if bidrectional:
            id = num * 2
            states_i = states[id: id+2, :, :]
        else:
            states_i = states[num, :, :]
            states_i = states_i[np.newaxis, :, :]
        return states_i


    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        op_type = "RNN"
        attr_dict = OrderedDict()
        # get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        #### get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]

        # get cur_node_layer node_index
        layer = node['node'].layer
        layer_name = layer.__class__.__name__


        # get layer attr
        input_size = layer.input_size
        hidden_size = layer.hidden_size
        num_layers = layer.num_layers
        bias = layer.bias
        batch_first = layer.batch_first
        # dropout = layer.dropout  # we don't need dropout  inference
        bidirectional = layer.bidirectional
        act = layer.mode[4:]
        states = layer.states
        # new_states = layer.new_states
        bidirect = 2 if bidirectional else 1

        #get layer weights
        weight_ih = layer.weight_ih
        weight_hh = layer.weight_hh
        bias_ih = None
        bias_hh = None
        if bias:
            bias_ih = layer.bias_ih
            bias_hh = layer.bias_hh

        attr_dict["direction"] = "bidirectional" if bidirectional else "forward"
        attr_dict["layout"] = 1 if batch_first else 0
        attr_dict["activations"] = [act.capitalize(), ] * bidirect
        attr_dict["hidden_size"] = hidden_size

        def name(num, name):
            return layer_name  + '_' + name + '_' + str(num)

        input = x_name
        for i in range(num_layers):
            w_i, r_i, b_i = cls.concat_params(i, weight_ih, weight_hh, bias_ih, bias_hh, bidirectional)
            attr_dict["inputs"] = [input]
            w_i_name = name(i, "w")
            attr_dict["inputs"].append(w_i_name)
            w_i_init = numpy_helper.from_array(w_i, w_i_name)
            onnx_init.append(w_i_init)
            r_i_name = name(i, 'r')
            attr_dict["inputs"].append(r_i_name)
            r_i_init = numpy_helper.from_array(r_i, r_i_name)
            onnx_init.append(r_i_init)
            if b_i is not None:
                b_i_name = name(i, 'b')
                attr_dict["inputs"].append(b_i_name)
                b_i_init = numpy_helper.from_array(b_i, b_i_name)
                onnx_init.append(b_i_init)
            else:
                attr_dict["inputs"].append("")
            # add sequence_lens into inputs
            if states is not None:
                state_i_name = name(i, 'h')
                attr_dict["inputs"].append("")
                attr_dict["inputs"].append(state_i_name)
                state_i = cls.concat_states(i, states, bidirectional)
                state_i_init = numpy_helper.from_array(state_i, state_i_name)
                onnx_init.append(state_i_init)

            attr_dict["outputs"] = [name(i, 'y')]
            rnn_node, y_out = make_node(op_type, **attr_dict)
            onnx_node.append(rnn_node)
            transpose_node, y_out_T = make_node("Transpose", inputs=[y_out], outputs=[y_out + "_T"], perm=[0,2,1,3])
            onnx_node.append(transpose_node)
            shape = np.array([0, 0, -1], dtype=np.int64)
            shape_name = name(i, 'shape')
            shape_value = numpy_helper.from_array(shape, shape_name)
            onnx_init.append(shape_value)
            if i + 1 < num_layers:
                reshape_output = [y_out + "_R"]
                reshape_node, y_out_R = make_node("Reshape", inputs=[y_out_T, shape_name], outputs=reshape_output)
                input = y_out_R
            else:
                reshape_node, y_out_R = make_node("Reshape", inputs=[y_out_T, shape_name], outputs=[out_name])
            onnx_node.append(reshape_node)

        return onnx_node, onnx_value, onnx_init



@OpMapper(["LSTM"])
class RNN():
    # suppport v1-v11

    @classmethod
    def concat_params(cls, num, weight_ih, weight_hh, bias_ih, bias_hh, bidrectional, hidden_size):

        def reform_weights(weights, hidden_size):
            reform_permutaion = [(0, 1), (3, 4), (1, 3)]
            slices = []
            for x, y in reform_permutaion:
                start = x * hidden_size
                end = y * hidden_size
                slices.append(weights[start:end])
            return np.concatenate(slices, axis=0)
        ih_i, hh_i, b_i = None, None, None
        if bidrectional:
            id = num * 2
            # get i-th rnn layer's weights - ih_i input to hidden
            ih_i_forward = weight_ih[id]
            ih_i_reverse = weight_ih[id + 1]
            ih_i_forward = tlx.convert_to_numpy(ih_i_forward)
            ih_i_reverse = tlx.convert_to_numpy(ih_i_reverse)
            ih_i_forward = reform_weights(ih_i_forward, hidden_size)
            ih_i_reverse = reform_weights(ih_i_reverse, hidden_size)
            ih_i_forward = ih_i_forward[np.newaxis, :, :]
            ih_i_reverse = ih_i_reverse[np.newaxis, :, :]
            ih_i = np.concatenate((ih_i_forward, ih_i_reverse), axis=0)

            # get i-th rnn layer's weights - hh_i hidden to hidden
            hh_i_forward = weight_hh[id]
            hh_i_reverse = weight_hh[id + 1]
            hh_i_forward = tlx.convert_to_numpy(hh_i_forward)
            hh_i_reverse = tlx.convert_to_numpy(hh_i_reverse)
            hh_i_forward = reform_weights(hh_i_forward, hidden_size)
            hh_i_reverse = reform_weights(hh_i_reverse, hidden_size)
            hh_i_forward = hh_i_forward[np.newaxis, :, :]
            hh_i_reverse = hh_i_reverse[np.newaxis, :, :]
            hh_i = np.concatenate((hh_i_forward, hh_i_reverse), axis=0)

            if bias_ih is not None:
                # get i-th rnn layer's bias - ih_b input to hidden
                b_ih_forward = bias_ih[id]
                b_ih_reverse = bias_ih[id + 1]
                b_ih_forward = tlx.convert_to_numpy(b_ih_forward)
                b_ih_reverse = tlx.convert_to_numpy(b_ih_reverse)
                b_ih_forward = reform_weights(b_ih_forward, hidden_size)
                b_ih_reverse = reform_weights(b_ih_reverse, hidden_size)
                b_ih_forward = b_ih_forward[np.newaxis, :]
                b_ih_reverse = b_ih_reverse[np.newaxis, :]
                # get i-th rnn layer's bias - hh_b hidden to hidden
                b_hh_forward = bias_hh[id]
                b_hh_reverse = bias_hh[id + 1]
                b_hh_forward = tlx.convert_to_numpy(b_hh_forward)
                b_hh_reverse = tlx.convert_to_numpy(b_hh_reverse)
                b_hh_forward = reform_weights(b_hh_forward, hidden_size)
                b_hh_reverse = reform_weights(b_hh_reverse, hidden_size)
                b_hh_forward = b_hh_forward[np.newaxis, :]
                b_hh_reverse = b_hh_reverse[np.newaxis, :]

                # concat bias
                b_forward = np.concatenate((b_ih_forward, b_hh_forward), axis=-1)
                b_reverse = np.concatenate((b_ih_reverse, b_hh_reverse), axis=-1)
                b_i = np.concatenate((b_forward, b_reverse), axis=0)
        else:
            # get i-th rnn layer's weights - ih_i input to hidden
            ih_i_forward = weight_ih[num]
            ih_i_forward = tlx.convert_to_numpy(ih_i_forward)
            ih_i_forward = reform_weights(ih_i_forward, hidden_size)
            ih_i = ih_i_forward[np.newaxis, :, :]

            # get i-th rnn layer's weights - hh_i hidden to hidden
            hh_i_forward = weight_hh[num]
            hh_i_forward = tlx.convert_to_numpy(hh_i_forward)
            hh_i_forward = reform_weights(hh_i_forward, hidden_size)
            hh_i = hh_i_forward[np.newaxis, :, :]

            if bias_ih is not None:
                # get i-th rnn layer's bias - ih_b input to hidden
                b_ih_forward = bias_ih[num]
                b_ih_forward = tlx.convert_to_numpy(b_ih_forward)
                b_ih_forward = reform_weights(b_ih_forward, hidden_size)
                b_ih_forward = b_ih_forward[np.newaxis, :]
                # get i-th rnn layer's bias - hh_b hidden to hidden
                b_hh_forward = bias_hh[num]
                b_hh_forward = tlx.convert_to_numpy(b_hh_forward)
                b_hh_forward = reform_weights(b_hh_forward, hidden_size)
                b_hh_forward = b_hh_forward[np.newaxis, :]

                # concat bias
                b_i = np.concatenate((b_ih_forward, b_hh_forward), axis=-1)

        return ih_i, hh_i, b_i

    @classmethod
    def concat_states(cls, num, states, bidrectional):
        states_h = tlx.convert_to_numpy(states[0])
        states_c = tlx.convert_to_numpy(states[1])
        if bidrectional:
            id = num * 2
            states_hi = states_h[id: id+2, :, :]
            states_ci = states_c[id: id+2, :, :]
        else:
            states_hi = states_h[num, :, :]
            states_hi = states_hi[np.newaxis, :, :]
            states_ci = states_c[num, :, :]
            states_ci = states_ci[np.newaxis, :, :]
        return states_hi, states_ci

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        op_type = "LSTM"
        attr_dict = OrderedDict()
        # get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        #### get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]

        # get cur_node_layer node_index
        layer = node['node'].layer
        layer_name = layer.__class__.__name__


        # get layer attr
        input_size = layer.input_size
        hidden_size = layer.hidden_size
        num_layers = layer.num_layers
        bias = layer.bias
        batch_first = layer.batch_first
        # dropout = layer.dropout  # we don't need dropout  inference
        bidirectional = layer.bidirectional
        states = layer.states
        # new_states = layer.new_states
        bidirect = 2 if bidirectional else 1

        #get layer weights
        weight_ih = layer.weight_ih
        weight_hh = layer.weight_hh
        bias_ih = None
        bias_hh = None
        if bias:
            bias_ih = layer.bias_ih
            bias_hh = layer.bias_hh

        attr_dict["direction"] = "bidirectional" if bidirectional else "forward"
        attr_dict["layout"] = 1 if batch_first else 0
        attr_dict["hidden_size"] = hidden_size
        attr_dict["activations"] = ['Sigmoid', 'Tanh', 'Tanh'] * bidirect
        attr_dict["input_forget"] = 0

        def name(num, name):
            return layer_name  + '_' + name + '_' + str(num)

        input = x_name
        for i in range(num_layers):
            w_i, r_i, b_i = cls.concat_params(i, weight_ih, weight_hh, bias_ih, bias_hh, bidirectional, hidden_size)
            attr_dict["inputs"] = [input]
            w_i_name = name(i, "w")
            attr_dict["inputs"].append(w_i_name)
            w_i_init = numpy_helper.from_array(w_i, w_i_name)
            onnx_init.append(w_i_init)
            r_i_name = name(i, 'r')
            attr_dict["inputs"].append(r_i_name)
            r_i_init = numpy_helper.from_array(r_i, r_i_name)
            onnx_init.append(r_i_init)
            if b_i is not None:
                b_i_name = name(i, 'b')
                attr_dict["inputs"].append(b_i_name)
                b_i_init = numpy_helper.from_array(b_i, b_i_name)
                onnx_init.append(b_i_init)
            else:
                attr_dict["inputs"].append("")
            # add sequence_lens into inputs
            if states is not None:
                state_hi_name = name(i, 'h')
                attr_dict["inputs"].append("")
                attr_dict["inputs"].append(state_hi_name)

                state_ci_name = name(i, 'c')
                attr_dict["inputs"].append(state_ci_name)
                state_hi, state_ci = cls.concat_states(i, states, bidirectional)
                state_hi_init = numpy_helper.from_array(state_hi, state_hi_name)
                onnx_init.append(state_hi_init)
                state_ci_init = numpy_helper.from_array(state_ci, state_ci_name)
                onnx_init.append(state_ci_init)

            attr_dict["outputs"] = [name(i, 'y')]
            rnn_node, y_out = make_node(op_type, **attr_dict)
            onnx_node.append(rnn_node)
            transpose_node, y_out_T = make_node("Transpose", inputs=[y_out], outputs=[y_out + "_T"], perm=[0,2,1,3])
            onnx_node.append(transpose_node)
            shape = np.array([0, 0, -1], dtype=np.int64)
            shape_name = name(i, 'shape')
            shape_value = numpy_helper.from_array(shape, shape_name)
            onnx_init.append(shape_value)
            if i + 1 < num_layers:
                reshape_output = [y_out + "_R"]
                reshape_node, y_out_R = make_node("Reshape", inputs=[y_out_T, shape_name], outputs=reshape_output)
                input = y_out_R
            else:
                reshape_node, y_out_R = make_node("Reshape", inputs=[y_out_T, shape_name], outputs=[out_name])
            onnx_node.append(reshape_node)

        return onnx_node, onnx_value, onnx_init