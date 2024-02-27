#!/usr/bin/env python3
from __future__ import annotations

import argparse
import enum
import faulthandler
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, TypeVar, cast

import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

if TYPE_CHECKING:
    from typing import TypeAlias

if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1)

NDArray: TypeAlias = 'np.ndarray[Any, Any]'

ARCH = gguf.MODEL_ARCH.LLAMA
ARCH_STR = gguf.MODEL_ARCH_NAMES[ARCH]

DEFAULT_CONCURRENCY = 8


@dataclass(frozen=True)
class DataType:
    name: str
    dtype: np.dtype[Any]
    valid_conversions: list[str]

    def elements_to_bytes(self, n_elements: int) -> int:
        return n_elements * self.dtype.itemsize


@dataclass(frozen=True)
class UnquantizedDataType(DataType):
    pass


@dataclass(frozen=True)
class QuantizedDataType(DataType):
    block_size: int
    quantized_dtype: np.dtype[Any]
    ggml_type: gguf.GGMLQuantizationType

    def quantize(self, arr: NDArray) -> NDArray:
        raise NotImplementedError(f'Quantization for {self.name} not implemented')

    def elements_to_bytes(self, n_elements: int) -> int:
        assert n_elements % self.block_size == 0, f'Invalid number of elements {n_elements} for {self.name} with block size {self.block_size}'
        return self.quantized_dtype.itemsize * (n_elements // self.block_size)


@dataclass(frozen=True)
class Q8_0QuantizedDataType(QuantizedDataType):
    # Mini Q8_0 quantization in Python!
    def quantize(self, arr: NDArray) -> NDArray:
        assert arr.size % self.block_size == 0 and arr.size != 0, f'Bad array size {arr.size}'
        assert arr.dtype == np.float32, f'Bad array type {arr.dtype}'
        n_blocks = arr.size // self.block_size
        blocks = arr.reshape((n_blocks, self.block_size))
        # Much faster implementation of block quantization contributed by @Cebtenzzre

        def quantize_blocks_q8_0(blocks: NDArray) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis = 1) / np.float32(127)
            with np.errstate(divide = 'ignore'):
                qs = (blocks / d[:, None]).round()
            qs[d == 0] = 0
            yield from zip(d, qs)
        return np.fromiter(quantize_blocks_q8_0(blocks), count = n_blocks, dtype = self.quantized_dtype)


DT_F16  = UnquantizedDataType('F16',  dtype = np.dtype(np.float16), valid_conversions = ['F32', 'Q8_0'])
DT_F32  = UnquantizedDataType('F32',  dtype = np.dtype(np.float32), valid_conversions = ['F16', 'Q8_0'])
DT_I32  = UnquantizedDataType('I32',  dtype = np.dtype(np.int16),   valid_conversions = [])
DT_BF16 = UnquantizedDataType('BF16', dtype = np.dtype(np.uint16),  valid_conversions = ['F32', 'F16', 'Q8_0'])

DT_Q8_0 = Q8_0QuantizedDataType('Q8_0',
                                dtype = np.dtype(np.float32), valid_conversions = [],
                                ggml_type = gguf.GGMLQuantizationType.Q8_0, block_size = 32,
                                quantized_dtype = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))]))


class GGMLFileType(enum.IntEnum):
    AllF32     = 0
    MostlyF16  = 1  # except 1d tensors
    MostlyQ8_0 = 7  # except 1d tensors

    def type_for_tensor(self, name: str, tensor: ReaderTensor) -> DataType:
        dt = GGML_FILE_TYPE_TO_DATA_TYPE.get(self)
        if dt is None:
            raise ValueError(self)
        # 1D tensors are always F32.
        return dt if len(tensor.shape) > 1 else DT_F32


GGML_QUANT_TYPE_TO_DATA_TYPE: dict[gguf.GGMLQuantizationType, DataType] = {
    gguf.GGMLQuantizationType.F32 : DT_F32,
    gguf.GGMLQuantizationType.F16 : DT_F16,
    gguf.GGMLQuantizationType.Q8_0 : DT_Q8_0,
}


GGML_FILE_TYPE_TO_DATA_TYPE: dict[GGMLFileType, DataType] = {
    GGMLFileType.AllF32    : DT_F32,
    GGMLFileType.MostlyF16 : DT_F16,
    GGMLFileType.MostlyQ8_0: DT_Q8_0,
}


@dataclass
class ReaderTensor:
    shape: list[int]
    data_type: DataType
    data: NDArray

    def __init__(self, idx, length, reader_tensor):
        self.shape = np.flip(reader_tensor.shape)
        self.data_type = GGML_QUANT_TYPE_TO_DATA_TYPE[reader_tensor.tensor_type]
        self.data = reader_tensor.data.reshape(self.shape)
        size = ' x '.join(f"{dim:6d}" for dim in reader_tensor.shape)
        padi = len(str(length))
        print(f"[{idx+1:{padi}d}/{length}] Reading tensor {reader_tensor.name:38s} | size {size:16} | type {self.data_type.name:4}")


def getDataFromReaderField(item):
    results = []
    for idx in item.data:
        results.append(item.parts[idx].tolist())
    try:
        results = np.array(results).squeeze()
    except Exception:
        results = results
    if len(item.types) == 1 and item.types[0] == gguf.GGUFValueType.STRING:
        results = ''.join([chr(int(c)) for c in results])
    elif len(item.types) == 2 and item.types[0] == gguf.GGUFValueType.ARRAY and item.types[1] == gguf.GGUFValueType.STRING:
        for i in range(len(results)):
            results[i] = ''.join([chr(int(c)) for c in results[i]])
    else:
        print(f"Types that no need to transfer: {item.types}")
    return results

@dataclass
class Params:
    # n_vocab:        int
    n_ctx:          int
    n_embd:         int
    n_layer:        int
    n_ff:           int
    n_head:         int
    n_head_kv:      int
    n_experts:      int | None = None
    n_experts_used: int | None = None
    f_norm_eps:     float | None = None

    f_rope_freq_base: float | None = None
    rope_scaling_type: gguf.RopeScalingType | None = None
    f_rope_scale: float | None = None
    n_orig_ctx: int | None = None
    rope_finetuned: bool | None = None

    ftype: GGMLFileType | None = None

    def __init__(self, fields) -> Params:
        self.n_embd = getDataFromReaderField(fields[f"{ARCH_STR}.embedding_length"])
        self.n_layer = getDataFromReaderField(fields[f"{ARCH_STR}.block_count"])
        self.n_ctx = getDataFromReaderField(fields[f"{ARCH_STR}.context_length"])
        self.n_ff = getDataFromReaderField(fields[f"{ARCH_STR}.feed_forward_length"])
        self.n_head = getDataFromReaderField(fields[f"{ARCH_STR}.attention.head_count"])
        self.n_head_kv = getDataFromReaderField(fields[f"{ARCH_STR}.attention.head_count_kv"])
        if f"{ARCH_STR}.expert_count" in fields:
            self.n_experts = getDataFromReaderField(fields[f"{ARCH_STR}.expert_count"])
        if f"{ARCH_STR}.expert_used_count" in fields:
            self.n_experts_used = getDataFromReaderField(fields[f"{ARCH_STR}.expert_used_count"])
        if f"{ARCH_STR}.attention.layer_norm_rms_epsilon" in fields:
            self.f_norm_eps = getDataFromReaderField(fields[f"{ARCH_STR}.attention.layer_norm_rms_epsilon"])
        if f"{ARCH_STR}.rope.freq_base" in fields:
            self.f_rope_freq_base = getDataFromReaderField(fields[f"{ARCH_STR}.rope.freq_base"])
        if f"{ARCH_STR}.rope.scaling.type" in fields:
            self.rope_scaling_type = getDataFromReaderField(fields[f"{ARCH_STR}.rope.scaling.type"])
        if f"{ARCH_STR}.rope.scaling.factor" in fields:
            self.f_rope_scale = getDataFromReaderField(fields[f"{ARCH_STR}.rope.scaling.factor"])
        if f"{ARCH_STR}.rope.scaling.original_context_length" in fields:
            self.n_orig_ctx = getDataFromReaderField(fields[f"{ARCH_STR}.rope.scaling.original_context_length"])
        if f"{ARCH_STR}.rope.scaling.finetuned" in fields:
            self.rope_finetuned = getDataFromReaderField(fields[f"{ARCH_STR}.rope.scaling.finetuned"])
        if "general.file_type" in fields:
            self.ftype = getDataFromReaderField(fields["general.file_type"])


class VocabLoader:
    def __init__(self, fields) -> None:
        self.vocab_type = getDataFromReaderField(fields["tokenizer.ggml.model"])
        self.texts = getDataFromReaderField(fields["tokenizer.ggml.tokens"])
        self.tokens = getDataFromReaderField(fields["tokenizer.ggml.scores"])
        self.toktypes = [gguf.TokenType(i) for i in getDataFromReaderField(fields["tokenizer.ggml.token_type"])]
        assert len(self.texts) == len(self.tokens)
        assert len(self.texts) == len(self.toktypes)
        self.vocab_size_base = len(self.texts)

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for i in range(self.vocab_size_base):
            yield self.texts[i], self.tokens[i], self.toktypes[i]

    def get_vocab_type(self) -> str:
        return self.vocab_type

    def __repr__(self) -> str:
        return f"<VocabLoader with {self.vocab_size_base} base tokens and {self.vocab_type} vocab type"


class SpecialVocabLoader:
    def __init__(self, fields):
        if "tokenizer.ggml.merges" in fields:
            self.merges = getDataFromReaderField(fields["tokenizer.ggml.merges"])
        else:
            self.merges = None

        def match(token, pattern):
            mat = re.match(pattern, token)
            if mat:
                return mat.group(1)
            else:
                return None

        self.special_token_ids = {}
        for key, value in fields.items():
            token = match(key, r"tokenizer\.ggml\.(.+)_(?:token_id)")
            if token:
                self.special_token_ids[token[:3]] = getDataFromReaderField(value)
        
        self.add_special_token = {}
        for key, value in fields.items():
            token = match(key, r"tokenizer\.ggml\.add_(.+)_(?:token)")
            if token:
                self.add_special_token[token[:3]] = getDataFromReaderField(value)
        
        if "tokenizer.chat_template" in fields:
            self.chat_template = getDataFromReaderField(fields["tokenizer.chat_template"])
        else:
            self.chat_template = None

    def add_to_gguf(self, gw: gguf.GGUFWriter, quiet: bool = False) -> None:
        if self.merges:
            if not quiet:
                print(f'gguf: Adding {len(self.merges)} merge(s).')
            gw.add_token_merges(self.merges)
        for typ, tokid in self.special_token_ids.items():
            id_handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if id_handler is None:
                print(
                    f'gguf: WARNING: No handler for special token type {typ} with id {tokid} - skipping',
                    file = sys.stderr,
                )
                continue
            if not quiet:
                print(f'gguf: Setting special token type {typ} to {tokid}')
            id_handler(tokid)
        for typ, value in self.add_special_token.items():
            add_handler: Callable[[bool], None] | None = getattr(gw, f'add_add_{typ}_token', None)
            if add_handler is None:
                print(
                    f'gguf: WARNING: No handler for add_{typ}_token with value {value} - skipping',
                    file = sys.stderr,
                )
                continue
            if not quiet:
                print(f'gguf: Setting add_{typ}_token to {value}')
            add_handler(value)
        if self.chat_template is not None:
            if not quiet:
                print(f'gguf: Setting chat_template to {self.chat_template}')
            gw.add_chat_template(self.chat_template)


def default_outfile(model_path: Path, file_type: GGMLFileType) -> Path:
    namestr = {
        GGMLFileType.AllF32:    "f32",
        GGMLFileType.MostlyF16: "f16",
        GGMLFileType.MostlyQ8_0:"q8_0",
    }[file_type]
    ret = model_path.parent / f"ggml-model-{namestr}-sorted.gguf"
    if ret == model_path:
        sys.stderr.write(
            f"Error: Default output path ({ret}) would overwrite the input. "
            "Please explicitly specify a path using --outfile.\n")
        sys.exit(1)
    return ret


def sort_layers(tensors):
    def get_order(x):
        if 'token_embd' in x:
            return -len(tensors)
        elif 'output' in x:
            return len(tensors)
        elif 'output_norm' in x:
            return len(tensors) + 1
        else:
            digit_idx = -1
            for i, item in enumerate(x):
                if np.char.isdigit(item):
                    digit_idx = i
            if digit_idx == -1:
                raise NotImplementedError(f"Unrecognized layer: {'.'.join(x)}")
            return int(x[digit_idx])

    sorted_tensors = {}
    for name, tensor in sorted(tensors.items(), key=lambda x: get_order(x[0].split('.'))):
        sorted_tensors[name] = tensor
    return sorted_tensors


Model: TypeAlias = 'dict[str, ReaderTensor]'


class OutputFile:
    def __init__(self, fname_out: Path, endianess:gguf.GGUFEndian = gguf.GGUFEndian.LITTLE) -> None:
        self.gguf = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[ARCH], endianess=endianess)

    def close(self) -> None:
        self.gguf.close()

    def add_meta_arch(self, params: Params) -> None:
        name = "LLaMA"

        # TODO: better logic to determine model name
        if params.n_ctx == 4096:
            name = "LLaMA v2"

        self.gguf.add_name                (name)
        self.gguf.add_context_length      (params.n_ctx)
        self.gguf.add_embedding_length    (params.n_embd)
        self.gguf.add_block_count         (params.n_layer)
        self.gguf.add_feed_forward_length (params.n_ff)
        self.gguf.add_rope_dimension_count(params.n_embd // params.n_head)
        self.gguf.add_head_count          (params.n_head)
        self.gguf.add_head_count_kv       (params.n_head_kv)

        if params.n_experts:
            self.gguf.add_expert_count(params.n_experts)

        if params.n_experts_used:
            self.gguf.add_expert_used_count(params.n_experts_used)

        if params.f_norm_eps:
            self.gguf.add_layer_norm_rms_eps(params.f_norm_eps)
        else:
            raise ValueError('f_norm_eps is None')

        if params.f_rope_freq_base is not None:
            self.gguf.add_rope_freq_base(params.f_rope_freq_base)

        if params.rope_scaling_type:
            assert params.f_rope_scale is not None
            self.gguf.add_rope_scaling_type(params.rope_scaling_type)
            self.gguf.add_rope_scaling_factor(params.f_rope_scale)

        if params.n_orig_ctx is not None:
            self.gguf.add_rope_scaling_orig_ctx_len(params.n_orig_ctx)

        if params.rope_finetuned is not None:
            self.gguf.add_rope_scaling_finetuned(params.rope_finetuned)

        if params.ftype is not None:
            self.gguf.add_file_type(params.ftype)

    def add_meta_vocab(self, vocab: VocabLoader) -> None:
        tokens = []
        scores = []
        toktypes = []
        # NOTE: `all_tokens` returns the base vocabulary and added tokens
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        vocab_type = vocab.get_vocab_type()
        self.gguf.add_tokenizer_model(vocab_type)
        self.gguf.add_token_list(tokens)
        self.gguf.add_token_scores(scores)
        self.gguf.add_token_types(toktypes)

    def add_meta_special_vocab(self, svocab: SpecialVocabLoader) -> None:
        svocab.add_to_gguf(self.gguf)

    def add_tensor_info(self, name: str, tensor: ReaderTensor) -> None:
        n_elements = int(np.prod(tensor.shape))
        raw_dtype = getattr(tensor.data_type, 'ggml_type', None)
        data_type = getattr(tensor.data_type, 'quantized_type', None) or tensor.data_type.dtype
        data_nbytes = tensor.data_type.elements_to_bytes(n_elements)
        self.gguf.add_tensor_info(name, tensor.shape, data_type, data_nbytes, raw_dtype = raw_dtype)

    def write_meta(self) -> None:
        self.gguf.write_header_to_file()
        self.gguf.write_kv_data_to_file()

    def write_tensor_info(self) -> None:
        self.gguf.write_ti_data_to_file()

    @staticmethod
    def write_all(
        fname_out: Path, ftype: GGMLFileType, params: Params, model: Model, vocab: VocabLoader, svocab: gguf.SpecialVocab,
        concurrency: int = DEFAULT_CONCURRENCY,
        endianess: gguf.GGUFEndian = gguf.GGUFEndian.LITTLE,
        pad_vocab: bool = False,
    ) -> None:
        of = OutputFile(fname_out, endianess=endianess)

        # meta data
        of.add_meta_arch(params)
        of.add_meta_vocab(vocab)
        of.add_meta_special_vocab(svocab)

        # tensor info
        for name, reader_tensor in model.items():
            of.add_tensor_info(name, reader_tensor)

        of.write_meta()
        of.write_tensor_info()

        start = time.time()
        for i, (name, reader_tensor) in enumerate(model.items()):
            elapsed = time.time() - start
            size = ' x '.join(f"{dim:6d}" for dim in reader_tensor.shape)
            padi = len(str(len(model)))
            print(f"[{i+1:{padi}d}/{len(model)}] Writing tensor {name:38s} | size {size:16} | type {reader_tensor.data_type.name:4} | T+{int(elapsed):4}")
            of.gguf.write_tensor_data(reader_tensor.data)

        of.close()


def main(args_in: list[str] | None = None) -> None:
    output_choices = ["f32", "f16"]
    if np.uint32(1) == np.uint32(1).newbyteorder("<"):
        # We currently only support Q8_0 output on little endian systems.
        output_choices.append("q8_0")
    parser = argparse.ArgumentParser(description="Convert a LLaMa model to a GGML compatible file")
    parser.add_argument("--outtype",     choices=output_choices, help="output format - note: q8_0 may be very slow (default: f16 or f32 based on input)")
    parser.add_argument("--outfile",     type=Path,              help="path to write to; default: based on input")
    parser.add_argument("model",         type=Path,              help="directory containing model file, or model file itself (*.pth, *.pt, *.bin)")
    parser.add_argument("--ctx",         type=int,               help="model training context (default: based on input)")
    parser.add_argument("--concurrency", type=int,               help=f"concurrency used for conversion (default: {DEFAULT_CONCURRENCY})", default = DEFAULT_CONCURRENCY)
    parser.add_argument("--bigendian",   action="store_true",    help="model is executed on big endian machine")
    parser.add_argument("--padvocab", action="store_true", help="add pad tokens when model vocab expects more than tokenizer metadata provides")

    args = parser.parse_args(args_in)
    
    assert args.model.suffix == '.gguf'
    gguf_src = gguf.GGUFReader(args.model)
    params = Params(gguf_src.fields)
    print(f"params = {params}")
    vocab = VocabLoader(gguf_src.fields)
    special_vocab = SpecialVocabLoader(gguf_src.fields)

    endianess = gguf.GGUFEndian.LITTLE
    if args.bigendian:
        endianess = gguf.GGUFEndian.BIG

    ftype   = params.ftype
    outfile = args.outfile or default_outfile(args.model, ftype)

    model = {}
    for i, reader_tensor in enumerate(gguf_src.tensors):
        name = reader_tensor.name
        model[name] = ReaderTensor(i, len(gguf_src.tensors), reader_tensor)

    model = sort_layers(model)

    OutputFile.write_all(outfile, ftype, params, model, vocab, special_vocab,
                         concurrency = args.concurrency, endianess = endianess, pad_vocab = args.padvocab)
    print(f"Wrote {outfile}")

if __name__ == '__main__':
    main()
