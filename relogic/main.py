from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np

import torch
from relogic.logickit.base import utils
from relogic.logickit.base.configure import configure, update_configure
from relogic.logickit.training import trainer, training_progress
from relogic.logickit.serving import Server
from relogic.logickit.analyzer.heads_importance import compute_heads_importance, mask_heads

if "PUDB" not in os.environ or os.environ["PUDB"] == "false":
  import relogic.utils.crash_on_ipy


def train(config):
  if config.use_external_teacher:
    teacher_model_path = config.teacher_model_path
    teacher_config = os.path.join(teacher_model_path, "general_config.json")
    with open(teacher_config) as f:
      teacher_config = SimpleNamespace(**json.load(f))
    teacher_config.local_rank = config.local_rank
    teacher_config.no_cuda = config.no_cuda
  else:
    teacher_config = None
  # A quick fix for loading external teacher
  model_trainer = trainer.Trainer(
    config=config, teacher_config=teacher_config)
  # A quick fix for version migration
  progress = training_progress.TrainingProgress(config=config)
  if config.use_external_teacher:
    model_path = os.path.join(teacher_model_path,
                              teacher_config.model_name + ".ckpt")
    model_trainer.restore(model_path)
    model_trainer.restore_teacher(model_path)
  model_trainer.train(progress)

def finetune(config):
  general_config_path = os.path.join(config.finetune_restore_path,
                                     "general_config.json")
  with open(general_config_path) as f:
    restore_config = SimpleNamespace(**json.load(f))
  if config.model_name:
    model_path = os.path.join(config.finetune_restore_path,
                            config.model_name + ".ckpt")
  else:
    model_path = os.path.join(config.finetune_restore_path,
                              restore_config.model_name + ".ckpt")

  model_trainer = trainer.Trainer(config)
  model_trainer.restore(model_path)
  progress = training_progress.TrainingProgress(config=config)
  model_trainer.train(progress)


def eval(config):
  general_config_path = os.path.join(config.restore_path,
                                     "general_config.json")
  with open(general_config_path) as f:
    restore_config = SimpleNamespace(**json.load(f))
  if config.model_name:
    model_path = os.path.join(config.restore_path,
                            config.model_name + ".ckpt")
  else:
    model_path = os.path.join(config.restore_path,
                              restore_config.model_name + ".ckpt")
  restore_config.mode = config.mode
  restore_config.local_rank = config.local_rank
  restore_config.no_cuda = config.no_cuda
  restore_config.buckets = config.buckets
  restore_config.gold_answer_file = config.gold_answer_file
  restore_config.null_score_diff_threshold = config.null_score_diff_threshold
  restore_config.output_attentions = config.output_attentions
  restore_config.use_external_teacher = False
  if not hasattr(restore_config, "branching_encoder"):
    restore_config.branching_encoder = False
  # Update the evaluation dataset
  update_configure(restore_config, config)
  print(restore_config)
  utils.heading("RUN {} ({:})".format(config.mode.upper(),
                                      restore_config.task_names))
  model_trainer = trainer.Trainer(restore_config)
  model_trainer.restore(model_path)
  if config.mode == "serving":
    server = Server(model_trainer)
    server.start()
  elif config.mode == "analysis":
    analyze(config, model_trainer)
  else:
    model_trainer.evaluate_all_tasks()

def analyze(config, model_trainer):
  # compute_heads_importance(config, model_trainer)
  mask_heads(config, model_trainer)


def main():
  utils.heading("SETUP")
  parser = argparse.ArgumentParser()

  # IO
  parser.add_argument(
    "--mode", default=None, choices=["train", "valid", "eval", "finetune", "analysis"])
  parser.add_argument("--output_dir", type=str, default="data/models")
  parser.add_argument("--max_seq_length", type=int, default=450)
  parser.add_argument("--max_query_length", type=int, default=64)
  parser.add_argument("--doc_stride", type=int, default=128)
  parser.add_argument("--do_lower_case", default=False, action="store_true")
  parser.add_argument("--model_name", type=str)
  parser.add_argument("--restore_path", type=str)
  parser.add_argument("--finetune_restore_path", type=str)
  parser.add_argument("--train_file", type=str, default="train.json")
  parser.add_argument("--dev_file", type=str, default="dev.json")
  parser.add_argument("--test_file", type=str, default="test.json")

  # Task Definition
  parser.add_argument("--task_names", type=str)
  parser.add_argument("--raw_data_path", type=str)
  parser.add_argument("--label_mapping_path", type=str)
  parser.add_argument("--unsupervised_data", type=str)
  parser.add_argument("--lang", type=str, default="en")
  parser.add_argument("--pretokenized", action="store_true", default=False)
  parser.add_argument("--topk", default=1)
  parser.add_argument("--gold_answer_file", default="data/preprocessed_data/squad20.json")
  parser.add_argument("--dump_to_files_dict", default="")

  parser.add_argument("--output_attentions", default=False, action="store_true")
  parser.add_argument("--span_inference", default=False, action="store_true")
  parser.add_argument("--metrics", default="", type=str)

  # Task related configuration

  # Sequence Labeling
  parser.add_argument("--sequence_labeling_use_cls", default=False, action="store_true")

  # Relation Extraction
  parser.add_argument("--no_entity_surface", dest="entity_surface_aware", default=True, action="store_false")
  parser.add_argument("--use_dependency_feature", dest="use_dependency_feature", default=False, action="store_true")
  parser.add_argument("--rel_extraction_module_type", type=str, default="hybrid")

  # Semantic Role Labeling
  parser.add_argument("--no_predicate_surface", dest="predicate_surface_aware", default=True, action="store_false")
  parser.add_argument("--no_span_annotation", dest="use_span_annotation", default=True, action="store_false")
  parser.add_argument("--use_span_candidates", default=False, action="store_true")
  parser.add_argument("--srl_module_type", type=str, default="sequence_labeling")
  parser.add_argument("--label_embed_dim", type=int, default=100)
  parser.add_argument("--external_vocab_embed_dim", type=int, default=300)
  parser.add_argument("--external_embeddings", type=str)
  parser.add_argument("--use_description", default=False, action="store_true")
  parser.add_argument("--srl_label_format", default="srl_label_span_based", type=str)
  parser.add_argument("--num_width_embeddings", type=int, default=300)
  parser.add_argument("--span_width_embedding_dim", type=int, default=100)
  parser.add_argument("--srl_candidate_loss", default=False, action="store_true")
  parser.add_argument("--srl_arg_span_repr", default="ave")
  parser.add_argument("--srl_pred_span_repr", default="ave")
  parser.add_argument("--srl_use_label_embedding", default=False, action="store_true")
  parser.add_argument("--srl_compute_pos_tag_loss", default=False, action="store_true")
  parser.add_argument("--srl_use_gold_predicate", default=False, action="store_true")
  parser.add_argument("--srl_use_gold_argument", default=False, action="store_true")

  # Dependency Parsing
  parser.add_argument("--dep_parsing_mlp_dim", default=300, type=int)
  parser.add_argument("--dropout", default=0.3, type=float)

  # Parallel Mapping
  parser.add_argument("--parallel_mapping_mode", default="alignment", type=str)


  # Reading Comprehension
  parser.add_argument("--null_score_diff_threshold", default=1.0)

  # Information Retrieval
  parser.add_argument("--qrels_file_path", type=str, default=None)

  # Modeling
  parser.add_argument("--use_gcn", dest="use_gcn", default=False, action="store_true")
  parser.add_argument("--fix_embedding", default=False, action="store_true")

  # Model
  parser.add_argument("--bert_model", type=str)
  parser.add_argument("--encoder_type", type=str, default="bert", choices=["bert", "xlm", "xlmr"])
  parser.add_argument("--hidden_size", type=int, default=768)
  parser.add_argument("--projection_size", type=int, default=300)
  parser.add_argument(
    "--initializer_range", type=float,
    default=0.02)  # initialization for task module
  # follow the initialization range of bert
  parser.add_argument("--no_bilstm", default=True, dest="use_bilstm", action="store_false")
  parser.add_argument("--repr_size", default=300, type=int)
  parser.add_argument("--branching_encoder", default=False, action="store_true")
  parser.add_argument("--routing_config_file", type=str)
  parser.add_argument("--selected_non_final_layers", type=str, default="none", help="split by ; among tasks")
  parser.add_argument("--dataset_type", type=str, default="bucket")
  parser.add_argument("--language_id_file", type=str, default=None)

  # Semi-Supervised
  parser.add_argument("--is_semisup", default=False, action="store_true")
  parser.add_argument("--partial_view_sources", type=str)
  parser.add_argument("--use_external_teacher", default=False, action="store_true")
  parser.add_argument("--teacher_model_path", default=None, type=str)

  # Training
  parser.add_argument("--seed", type=int, default=3435)
  parser.add_argument("--no_cuda", action="store_true")
  parser.add_argument("--local_rank", type=int, default=-1)
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  parser.add_argument("--warmup_proportion", type=float, default=0.1)
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help=
    "Number of updates steps to accumulate before performing a backward/update pass"
  )
  parser.add_argument("--print_every", type=int, default=25)
  parser.add_argument("--eval_dev_every", default=2000, type=int)
  parser.add_argument("--train_batch_size", type=str, default="8")
  parser.add_argument("--test_batch_size", type=str, default="8")
  parser.add_argument("--grad_clip", type=float, default=1.0)
  parser.add_argument("--epoch_number", type=int, default=20)
  parser.add_argument("--self_attention_head_size", default=64, type=int)
  parser.add_argument("--schedule_method", default="warmup_linear")
  parser.add_argument(
    "--no_schedule_lr", dest="schedule_lr", default=True, action="store_false")
  parser.add_argument("--word_dropout", default=False, action="store_true")
  parser.add_argument("--word_dropout_prob", default=0.6, type=float)
  parser.add_argument("--max_margin", type=float, default=3)
  parser.add_argument("--warmup_epoch_number", type=int, default=0)
  parser.add_argument("--sgd_learning_rate", type=float, default=0.1)
  parser.add_argument("--adam_learning_rate", type=float, default=0.001)
  parser.add_argument("--sep_optim", dest="sep_optim", default=False, action="store_true")
  parser.add_argument("--multi_gpu", dest="multi_gpu", default=False, action="store_true")
  parser.add_argument("--ignore_parameters", default="", type=str)
  parser.add_argument("--fix_bert", default=False, action="store_true")
  parser.add_argument("--two_stage_optim", default=False, action="store_true")
  parser.add_argument("--training_scheme", default=None, type=str)
  parser.add_argument("--training_scheme_file", default=None, type=str)
  parser.add_argument("--num_train_optimization_steps", default=0, type=int)
  parser.add_argument("--early_stop_at", default=0, type=int)
  parser.add_argument("--loss_weight", type=str, default='1')
  parser.add_argument("--select_index_method", type=str, default="cls")
  parser.add_argument("--use_cosine_loss", default=False, action="store_true")
  parser.add_argument("--adversarial_training", default=None, type=str)
  # We allow to set same training steps for different dataset
  # Need to combine to CUDA_VISIBLE_DEVICES

  # Analysis
  parser.add_argument("--head_to_mask_file", type=str, default="")


  # Configuration
  parser.add_argument("--config_file", type=str, default=None)
  parser.add_argument("--trainer_config", type=str, default=None)
  parser.add_argument("--module_config", type=str, default=None)
  parser.add_argument("--task_config", type=str, default=None)

  args = parser.parse_args()

  if not args.mode:
    raise ValueError("You need to specify the mode")
  if args.output_dir:
    if os.path.exists(args.output_dir) and os.listdir(
        args.output_dir) and args.mode == "train":
      raise ValueError(
        "Output directory ({}) already exists and is not empty.".format(
          args.output_dir))
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

  if args.gradient_accumulation_steps < 1:
    raise ValueError(
      "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".
      format(args.gradient_accumulation_steps))
  # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
  # num_train_optimization_steps = len(train_examples) / batch_size * epoch_number

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  configure(args)

  print(args)

  if args.mode == "train":
    utils.heading("START TRAINING ({:})".format(args.task_names))
    train(args)
  elif args.mode == "valid":
    eval(args)
  elif args.mode == "eval":
    eval(args)
  elif args.mode == "finetune":
    finetune(args)
  elif args.mode == "serving":
    eval(args)
  elif args.mode == "analysis":
    eval(args)


if __name__ == "__main__":
  main()
