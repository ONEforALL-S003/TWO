# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import tempfile
import onnx
import onnx_tf

# ONNX legalizer is an optional feature
# It enables conversion of some operations, but in experimental phase for now
try:
    import onnx_legalizer
    _onnx_legalizer_enabled = True
except ImportError:
    _onnx_legalizer_enabled = False

import onelib.make_cmd as _make_cmd
import onelib.utils as oneutils

# TODO Find better way to suppress trackback on error
sys.tracebacklimit = 0


# Class to rename input/output to prevent issues while import ONNX models
class TidyIONames:
    def __init__(self, onnx_model):
        self.input_nodes = []
        self.output_nodes = []
        self.remap_inputs = []
        self.remap_outputs = []
        self.initializers = []
        self.onnx_model = onnx_model
        # some models may have initializers as inputs. ignore them.
        for initializer in onnx_model.graph.initializer:
            self.initializers.append(initializer.name)

    def order(self):
        for idx in range(0, len(self.onnx_model.graph.input)):
            name = self.onnx_model.graph.input[idx].name
            if not name in self.initializers:
                self.input_nodes.append(name)
                self.remap_inputs.append('i_' + format(idx + 1, '04d') + '_' + name)
        for idx in range(0, len(self.onnx_model.graph.output)):
            name = self.onnx_model.graph.output[idx].name
            self.output_nodes.append(name)
            self.remap_outputs.append('o_' + format(idx + 1, '04d') + '_' + name)

    # exclude special characters in names
    def sanitize(self):
        for idx in range(0, len(self.onnx_model.graph.input)):
            name = self.onnx_model.graph.input[idx].name
            if not name in self.initializers:
                if '.' in name or ':' in name or name[:1].isdigit():
                    self.input_nodes.append(name)
                    name_alt = name.replace('.', '_')
                    name_alt = name_alt.replace(':', '_')
                    if name_alt[:1].isdigit():
                        name_alt = 'a_' + name_alt
                    self.remap_inputs.append(name_alt)
        for idx in range(0, len(self.onnx_model.graph.output)):
            name = self.onnx_model.graph.output[idx].name
            if '.' in name or ':' in name or name[:1].isdigit():
                self.output_nodes.append(name)
                name_alt = name.replace('.', '_')
                name_alt = name_alt.replace(':', '_')
                if name_alt[:1].isdigit():
                    name_alt = 'a_' + name_alt
                self.remap_outputs.append(name_alt)

    def update(self):
        # change names for graph input
        for i in range(len(self.onnx_model.graph.input)):
            if self.onnx_model.graph.input[i].name in self.input_nodes:
                to_rename = self.onnx_model.graph.input[i].name
                idx = self.input_nodes.index(to_rename)
                self.onnx_model.graph.input[i].name = self.remap_inputs[idx]
        # change names of all nodes in the graph
        for i in range(len(self.onnx_model.graph.node)):
            # check node.input is to change to remap_inputs or remap_outputs
            for j in range(len(self.onnx_model.graph.node[i].input)):
                if self.onnx_model.graph.node[i].input[j] in self.input_nodes:
                    to_rename = self.onnx_model.graph.node[i].input[j]
                    idx = self.input_nodes.index(to_rename)
                    self.onnx_model.graph.node[i].input[j] = self.remap_inputs[idx]
                if self.onnx_model.graph.node[i].input[j] in self.output_nodes:
                    to_rename = self.onnx_model.graph.node[i].input[j]
                    idx = self.output_nodes.index(to_rename)
                    self.onnx_model.graph.node[i].input[j] = self.remap_outputs[idx]
            # check node.output is to change to remap_inputs or remap_outputs
            for j in range(len(self.onnx_model.graph.node[i].output)):
                if self.onnx_model.graph.node[i].output[j] in self.output_nodes:
                    to_rename = self.onnx_model.graph.node[i].output[j]
                    idx = self.output_nodes.index(to_rename)
                    self.onnx_model.graph.node[i].output[j] = self.remap_outputs[idx]
                if self.onnx_model.graph.node[i].output[j] in self.input_nodes:
                    to_rename = self.onnx_model.graph.node[i].output[j]
                    idx = self.input_nodes.index(to_rename)
                    self.onnx_model.graph.node[i].output[j] = self.remap_inputs[idx]
        # change names for graph output
        for i in range(len(self.onnx_model.graph.output)):
            if self.onnx_model.graph.output[i].name in self.output_nodes:
                to_rename = self.onnx_model.graph.output[i].name
                idx = self.output_nodes.index(to_rename)
                self.onnx_model.graph.output[i].name = self.remap_outputs[idx]


def get_driver_cfg_section():
    return "one-import-onnx"


def _get_parser():
    parser = argparse.ArgumentParser(
        description='command line tool to convert ONNX to circle')

    oneutils.add_default_arg(parser)

    ## tf2tfliteV2 arguments
    tf2tfliteV2_group = parser.add_argument_group('converter arguments')

    # input and output path.
    tf2tfliteV2_group.add_argument(
        '-i', '--input_path', type=str, help='full filepath of the input file')
    tf2tfliteV2_group.add_argument(
        '-o', '--output_path', type=str, help='full filepath of the output file')

    # input and output arrays.
    tf2tfliteV2_group.add_argument(
        '-I',
        '--input_arrays',
        type=str,
        help='names of the input arrays, comma-separated')
    tf2tfliteV2_group.add_argument(
        '-O',
        '--output_arrays',
        type=str,
        help='names of the output arrays, comma-separated')

    # fixed options
    tf2tfliteV2_group.add_argument('--model_format', default='saved_model')
    tf2tfliteV2_group.add_argument('--converter_version', default='v2')

    parser.add_argument('--unroll_rnn', action='store_true', help='Unroll RNN operators')
    parser.add_argument(
        '--unroll_lstm', action='store_true', help='Unroll LSTM operators')
    parser.add_argument(
        '--keep_io_order',
        action='store_true',
        help=
        'Ensure generated circle model preserves the I/O order of the original onnx model.'
    )

    # save intermediate file(s)
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        help='Save intermediate files to output folder')

    # experimental options
    parser.add_argument(
        '--experimental_disable_batchmatmul_unfold',
        action='store_true',
        help='Experimental disable BatchMatMul unfold')

    return parser


def _verify_arg(parser, args):
    """verify given arguments"""
    # check if required arguments is given
    missing = []
    if not oneutils.is_valid_attr(args, 'input_path'):
        missing.append('-i/--input_path')
    if not oneutils.is_valid_attr(args, 'output_path'):
        missing.append('-o/--output_path')
    if len(missing):
        parser.error('the following arguments are required: ' + ' '.join(missing))


def _parse_arg(parser):
    args = parser.parse_args()
    # print version
    if args.version:
        oneutils.print_version_and_exit(__file__)

    return args


def _apply_verbosity(verbosity):
    # NOTE
    # TF_CPP_MIN_LOG_LEVEL
    #   0 : INFO + WARNING + ERROR + FATAL
    #   1 : WARNING + ERROR + FATAL
    #   2 : ERROR + FATAL
    #   3 : FATAL
    if verbosity:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# TF2.12.1 tries to sanitize special characters, '.:' and maybe others and then fails with
# 'IndexError: tuple index out of range' error from somewhere else.
# This method is to prevent this IndexError.
def _sanitize_io_names(onnx_model):
    sanitizer = TidyIONames(onnx_model)
    sanitizer.sanitize()
    sanitizer.update()


# The index of input/output is added in front of the name. For example,
# Original input names: 'a', 'c', 'b'
# Renamed: 'i_0001_a', 'i_0002_c', 'i_0003_b'
# This will preserve I/O order after import.
def _remap_io_names(onnx_model):
    # gather existing name of I/O and generate new name of I/O in sort order
    remapper = TidyIONames(onnx_model)
    remapper.order()
    remapper.update()


def _check_ext():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ext_path = os.path.join(dir_path, 'one-import-onnx-ext')
    if (os.path.isfile(ext_path)):
        return ext_path
    return None


def _convert(args):
    _apply_verbosity(args.verbose)

    # get file path to log
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile_path = os.path.realpath(args.output_path) + '.log'
    ext_path = _check_ext()

    with open(logfile_path, 'wb') as f, tempfile.TemporaryDirectory() as tmpdir:
        # save intermediate
        if oneutils.is_valid_attr(args, 'save_intermediate'):
            tmpdir = os.path.dirname(logfile_path)
        # convert onnx to tf saved model
        onnx_model = onnx.load(getattr(args, 'input_path'))
        _sanitize_io_names(onnx_model)
        if _onnx_legalizer_enabled:
            options = onnx_legalizer.LegalizeOptions
            options.unroll_rnn = oneutils.is_valid_attr(args, 'unroll_rnn')
            options.unroll_lstm = oneutils.is_valid_attr(args, 'unroll_lstm')
            onnx_legalizer.legalize(onnx_model, options)
        if oneutils.is_valid_attr(args, 'keep_io_order'):
            _remap_io_names(onnx_model)
            if oneutils.is_valid_attr(args, 'save_intermediate'):
                basename = os.path.basename(getattr(args, 'input_path'))
                fixed_path = os.path.join(tmpdir,
                                          os.path.splitext(basename)[0] + '~.onnx')
                onnx.save(onnx_model, fixed_path)

        if ext_path:
            # save onnx_model to temporary alt file
            basename = os.path.basename(getattr(args, 'input_path'))
            alt_path = os.path.join(tmpdir, os.path.splitext(basename)[0] + '-alt.onnx')
            onnx.save(onnx_model, alt_path)

            # call extension with options
            ext_cmd = [ext_path]
            if oneutils.is_valid_attr(args, 'unroll_rnn'):
                ext_cmd.append('--unroll_rnn')
            if oneutils.is_valid_attr(args, 'unroll_lstm'):
                ext_cmd.append('--unroll_lstm')
            if oneutils.is_valid_attr(args, 'experimental_disable_batchmatmul_unfold'):
                ext_cmd.append('--experimental_disable_batchmatmul_unfold')
            if oneutils.is_valid_attr(args, 'save_intermediate'):
                ext_cmd.append('--save_intermediate')
            if oneutils.is_valid_attr(args, 'keep_io_order'):
                ext_cmd.append('--keep_io_order')
            ext_cmd.append(alt_path)
            ext_cmd.append(getattr(args, 'output_path'))
            oneutils.run(ext_cmd, logfile=f)
            return

        tf_savedmodel = onnx_tf.backend.prepare(onnx_model)

        savedmodel_name = os.path.splitext(os.path.basename(
            args.output_path))[0] + '.savedmodel'
        savedmodel_output_path = os.path.join(tmpdir, savedmodel_name)
        tf_savedmodel.export_graph(savedmodel_output_path)

        # make a command to convert from tf to tflite
        tf2tfliteV2_path = os.path.join(dir_path, 'tf2tfliteV2.py')
        tf2tfliteV2_output_name = os.path.splitext(os.path.basename(
            args.output_path))[0] + '.tflite'
        tf2tfliteV2_output_path = os.path.join(tmpdir, tf2tfliteV2_output_name)

        tf2tfliteV2_cmd = _make_cmd.make_tf2tfliteV2_cmd(
            args, tf2tfliteV2_path, savedmodel_output_path, tf2tfliteV2_output_path)

        f.write((' '.join(tf2tfliteV2_cmd) + '\n').encode())

        # convert tf to tflite
        oneutils.run(tf2tfliteV2_cmd, logfile=f)

        # make a command to convert from tflite to circle
        tflite2circle_path = os.path.join(dir_path, 'tflite2circle')
        tflite2circle_cmd = _make_cmd.make_tflite2circle_cmd(tflite2circle_path,
                                                             tf2tfliteV2_output_path,
                                                             getattr(args, 'output_path'))

        f.write((' '.join(tflite2circle_cmd) + '\n').encode())

        # convert tflite to circle
        oneutils.run(tflite2circle_cmd, err_prefix="tflite2circle", logfile=f)


def main():
    # parse arguments
    parser = _get_parser()
    args = _parse_arg(parser)

    # parse configuration file
    oneutils.parse_cfg(args.config, 'one-import-onnx', args)

    # verify arguments
    _verify_arg(parser, args)

    # convert
    _convert(args)


if __name__ == '__main__':
    oneutils.safemain(main, __file__)
