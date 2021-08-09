import copy
from typing import Sequence
from itertools import product

import torch
import pandas as pd

from ..utils import as_list, is_sequence


class LabeledTensor:
    def __init__(
            self,
            dim_names: Sequence[str],
            dim_keys: Sequence[Sequence[str]]
    ):
        if len(dim_names) != len(dim_keys):
            raise ValueError(f"The number of dimension names ({len(dim_names)}) "
                             f"does not match the number of dimension keys ({len(dim_keys)}")
        self.dim_names = dim_names
        self.dim_keys = dim_keys

        self.dim_key_map = [
            {key: i for i, key in enumerate(keys)}
            for keys in dim_keys
        ]

        shape = [len(keys) for keys in dim_keys]
        self.data = torch.zeros(shape)

    def parse_key(self, key):
        key = as_list(key)

        if any(k is Ellipsis for k in key):
            raise NotImplementedError("Elipsis indexing is not supported for LabeledTensors")

        for i, k in enumerate(key):
            key_map = self.dim_key_map[i]
            if isinstance(k, str):
                key[i] = key_map[k]
            elif is_sequence(k):
                key[i] = [key_map[elem] if isinstance(elem, str) else elem for elem in k]

        return tuple(key)

    def __getitem__(self, key) -> torch.Tensor:
        key = self.parse_key(key)
        return self.data[key]

    def __setitem__(self, key, value):
        key = self.parse_key(key)
        self.data[key] = value

    def to_dataframe(self):
        df_dict = {dim: [] for dim in self.dim_names[:-1]}
        df_dict.update({dim: [] for dim in self.dim_keys[-1]})

        for keys in product(*self.dim_keys[:-1]):
            for dim, key in zip(self.dim_names[:-1], keys):
                df_dict[dim].append(key)
            for dim, value in zip(self.dim_keys[-1], self[keys].tolist()):
                df_dict[dim].append(value)
        df = pd.DataFrame(df_dict)
        return df

    def to_dict(self):
        nested_dict = 0
        for keys in reversed(self.dim_keys):
            nested_dict = {key: copy.deepcopy(nested_dict) for key in keys}

        for key in product(*self.dim_keys):
            value = self[key].item()
            set_dict = nested_dict
            for k in key[:-1]:
                set_dict = set_dict[k]
            set_dict[key[-1]] = value

        return nested_dict

    def compute_summary_stats(self, summary_stats_to_output):
        summary_stats = LabeledTensor(dim_names=["summary_stat", *self.dim_names[1:]],
                                      dim_keys=[summary_stats_to_output, *self.dim_keys[1:]])
        summary_stat_funcs = LabeledTensor.get_summary_stat_funcs()

        for keys in product(*self.dim_keys[1:]):
            values = self[(slice(None), *keys)]
            for summary_stat_name in summary_stats_to_output:
                summary_stat_func = summary_stat_funcs[summary_stat_name]
                summary_stat = summary_stat_func(values).item()
                summary_stats[(summary_stat_name, *keys)] = summary_stat

        return summary_stats

    @staticmethod
    def fix_tensor(x):
        x = x[x.isfinite()]
        if x.shape[0] == 0:
            return torch.tensor([0.])
        return x

    @staticmethod
    def get_summary_stat_funcs(dim: int = 0):
        return {
            'mean': lambda x: torch.mean(LabeledTensor.fix_tensor(x), dim=dim),
            'median': lambda x: torch.median(LabeledTensor.fix_tensor(x), dim=dim).values,
            'mode': lambda x: torch.mode(LabeledTensor.fix_tensor(x), dim=dim).values,
            'std': lambda x: torch.std(LabeledTensor.fix_tensor(x), dim=dim),
            'min': lambda x: torch.min(LabeledTensor.fix_tensor(x), dim=dim).values,
            'max': lambda x: torch.max(LabeledTensor.fix_tensor(x), dim=dim).values,
        }
