#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import inspect
import logging

OP_MAPPING_NO_REGISTER = 0
OP_MAPPING_NO_VERSION = 1
OP_MAPPING_SUCCESSED = 2
OP_MAPPING_FAILED = 3


def get_max_support_version(versions, opset_version):
    max_version = -1
    for vs in sorted(versions):
        if vs <= opset_version:
            max_version = vs
    return max_version

class OpMapper(object):
    OPSETS = {}
    # TODO: CUSTOM_OP = {}
    def __init__(self, tlx_op, **kwargs):
        if not isinstance(tlx_op, list):
            tlx_op = [tlx_op]
        self.tlx_op = tlx_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("version_"):
                version = int(k.replace("version_", ""))
                for op in self.tlx_op:
                    if op not in OpMapper.OPSETS:
                        OpMapper.OPSETS[op] = {}
                    opset_dict = OpMapper.OPSETS[op]
                    opset_dict[version] = (v, self.kwargs)

    @staticmethod
    def mapping(node, opset_version):
        """

        Parameters
        ----------
        node : tlx_node
            tlx_node
        opset_version : int
            the version of onnx_op to convert

        Returns
        -------

        """
        try:
            # TODO : if node.layer.__class__.__name__ in CUSTOM_OP
            node_type = node['node'].layer.__class__.__name__
            opsets = OpMapper.OPSETS[node_type]
            versions = list(opsets.keys())
            convert_verison = get_max_support_version(versions, opset_version)
            mapper_func, kw= opsets[convert_verison]
            return mapper_func(node, **kw)
        except Exception as e:
            raise Exception(
                "Unsupported mapping node [{}] to onnx node, which op_type is {}, specific error: .".
                    format(node['node'].layer, node['node'].layer.__class__.__name__) + str(e)
            )

    @staticmethod
    def update_opset_version(graph, opset_version):
        recommend_opset_version = OpMapper.check_support_version(
            graph, opset_version, True
        )
        # TODO : CUSTOM OP CHECK
        # for tlx_node_list in graph:
        #     for tlx_node in tlx_node_list:
        #         pass
        if opset_version != recommend_opset_version:
            warning_info = "\n======================\n"
            warning_info += "\nFor a successful conversion, set the recommended opset version : {}\n".format(
                recommend_opset_version)
            warning_info += "\n======================\n"
            logging.warning(warning_info)
        return recommend_opset_version

    @staticmethod
    def check_support_version(graph, opset_version, for_check = False):
        op_mapping_status = {
            OP_MAPPING_NO_REGISTER: [],
            OP_MAPPING_NO_VERSION: [],
        }
        for key in graph.keys():
            tlx_node = graph[key]["node"]
            # TODO : CUSTOM OP CHECK
            if tlx_node.layer.__class__.__name__ in ['Input', '_InputLayer']:
                continue
            node_type = tlx_node.layer.__class__.__name__
                # check act_type
            if hasattr(tlx_node.layer, "act") and tlx_node.layer.act != None:
                act_type = tlx_node.layer.act.__class__.__name__
                if act_type not in OpMapper.OPSETS:
                    op_mapping_status[OP_MAPPING_NO_REGISTER].append(node_type)
                else:
                    opsets = OpMapper.OPSETS[act_type]
                    versions = list(opsets.keys())
                    convert_version = get_max_support_version(versions, opset_version)
                    if convert_version == -1:
                        op_mapping_status[OP_MAPPING_NO_VERSION].append(act_type)

            # check node_type
            if node_type not in OpMapper.OPSETS:
                op_mapping_status[OP_MAPPING_NO_REGISTER].append(node_type)
            else:
                opsets = OpMapper.OPSETS[node_type]
                versions = list(opsets.keys())
                convert_version = get_max_support_version(versions, opset_version)
                if convert_version == -1:
                    op_mapping_status[OP_MAPPING_NO_VERSION].append(node_type)

        if len(op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
            unsupported_op_types = set(op_mapping_status[OP_MAPPING_NO_REGISTER])
            error_info = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_op_types))
            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

        if len(op_mapping_status[OP_MAPPING_NO_VERSION]) > 0:
            unsupported_op_types = set(op_mapping_status[OP_MAPPING_NO_VERSION])
            recommend_opset_version = -1
            for op_type in unsupported_op_types:
                opsets = OpMapper.OPSETS[op_type]
                if min(opsets.keys()) > recommend_opset_version:
                    recommend_opset_version = min(opsets.keys())
            warning_info = "\nThere are {} ops that are not supported in opset version {}, please set opset version >= {}.\n".format(
                len(unsupported_op_types), opset_version,
                recommend_opset_version)
            for op_type in unsupported_op_types:
                warning_info += "=========== {} ===========\n".format(op_type)
            if for_check:
                logging.warning(warning_info)
                return recommend_opset_version
            raise NotImplementedError(warning_info)

        return opset_version