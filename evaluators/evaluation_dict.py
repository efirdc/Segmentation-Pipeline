import copy
from typing import Sequence, Dict
from itertools import product

import torch
import pandas as pd


class EvaluationDict:
    def __init__(self, dimensions: Sequence[str], dimension_keys: Sequence[Sequence[str]], initial_value=None):
        self.dimensions = dimensions
        self.dimension_keys = dimension_keys
        self.nested_dict = initial_value
        for keys in reversed(dimension_keys):
            self.nested_dict = {key: copy.deepcopy(self.nested_dict) for key in keys}

    def __getitem__(self, keys: Sequence[str]):
        value = self.nested_dict
        for key in keys:
            value = value[key]
        return value

    def __setitem__(self, keys, value):
        nested_dict = self.nested_dict
        for key in keys[:-1]:
            nested_dict = nested_dict[key]
        nested_dict[keys[-1]] = value

    def to_dataframe(self):
        df_dict = {dim: [] for dim in self.dimensions[:-1]}
        df_dict.update({dim: [] for dim in self.dimension_keys[-1]})

        for keys in product(*self.dimension_keys[:-1]):
            for dim, key in zip(self.dimensions[:-1], keys):
                df_dict[dim].append(key)
            for dim, value in self[keys].items():
                df_dict[dim].append(value)

        df = pd.DataFrame(df_dict)
        return df

    def compute_summary_stats(self, summary_stats_to_output):
        summary_stats = EvaluationDict(dimensions=["summary_stat", *self.dimensions[1:]],
                                       dimension_keys=[summary_stats_to_output, *self.dimension_keys[1:]])
        summary_stat_funcs = EvaluationDict.get_summary_stat_funcs()

        first_dim_keys = self.dimension_keys[0]
        for keys in product(*self.dimension_keys[1:]):
            values = [self[(first_dim_key, *keys)] for first_dim_key in first_dim_keys]

            for summary_stat_name in summary_stats_to_output:
                summary_stat_func = summary_stat_funcs[summary_stat_name]
                summary_stat = summary_stat_func(torch.tensor(values).float()).item()
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
            'mean': lambda x: torch.mean(EvaluationDict.fix_tensor(x), dim=dim),
            'median': lambda x: torch.median(EvaluationDict.fix_tensor(x), dim=dim).values,
            'mode': lambda x: torch.mode(EvaluationDict.fix_tensor(x), dim=dim).values,
            'std': lambda x: torch.std(EvaluationDict.fix_tensor(x), dim=dim),
            'min': lambda x: torch.min(EvaluationDict.fix_tensor(x), dim=dim).values,
            'max': lambda x: torch.max(EvaluationDict.fix_tensor(x), dim=dim).values,
        }

    def __repr__(self):
        return self.nested_dict.__repr__()

    def __str__(self):
        return self.nested_dict.__str__()
