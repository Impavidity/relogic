#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import os
import copy
import numpy as np

from relogic.utils import tool
from relogic.utils.configue import Configure


class EvaluateTool(object):
    """
    The meta evaluator
    """
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, preds, golds, section):
        meta_args = self.meta_args
        summary = {}
        wait_for_eval = {}

        for pred, gold in zip(preds, golds):
            if gold['arg_path'] not in wait_for_eval.keys():
                wait_for_eval[gold['arg_path']] = {'preds': [], "golds":[]}
            wait_for_eval[gold['arg_path']]['preds'].append(pred)
            wait_for_eval[gold['arg_path']]['golds'].append(gold)

        for arg_path, preds_golds in wait_for_eval.items():
            args = Configure.refresh_args_by_file_cfg(os.path.join(meta_args.dir.configure, arg_path), meta_args)
            evaluator = tool.get_evaluator(args.evaluate.tool)(args)
            summary_tmp = evaluator.evaluate(preds_golds['preds'], preds_golds['golds'], section)
            for key, metric in summary_tmp.items():  # TODO
                summary[os.path.join(arg_path, key)] = metric
            # summary[os.path.join(arg_path, args.train.stop)] = summary_tmp[args.train.stop]

        summary['avr'] = float(np.mean([float(v) for k, v in summary.items()]))
        return summary
