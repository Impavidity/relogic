class TextToSQLArguments:
  @classmethod
  def add_task_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--text_to_sql_model_type",
      type=str,
      choices=["column_selection"],
      help="Choose the subtask for the text to sql.",
    )

  @classmethod
  def add_model_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--column_selection_method",
      type=str,
      choices=["_TABLE_"],
      help="The argument will determine the method for aggregating column representation for column selection.",
    )


class EditNetArguments:
  @classmethod
  def add_task_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--use_lr_scheduler",
      default=False,
      action="store_true"
    )

  @classmethod
  def add_model_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--editnet_action_embedding_dim",
      type=int,
      default=128)
    parser.add_argument(
      "--editnet_hidden_size",
      type=int,
      default=300)
    parser.add_argument(
      "--editnet_schema_embedding_dim",
      type=int,
      default=768)
    parser.add_argument(
      "--editnet_bert_embedding_dim",
      type=int,
      default=768
    )



class RATSQLArguments:
  @classmethod
  def add_task_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--use_lr_scheduler",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--lr_scheduler_gammar",
      default=0.5
    )
    parser.add_argument(
      "--action_candidate_space",
      type=str,
      choices=["all", "constrained"],
      help="At each step, we use all the actions or use constrained action space",
      default="constrained"
    )
    parser.add_argument(
      "--use_deliberate_decoding",
      default=False,
      action="store_true")
    parser.add_argument(
      "--use_deliberate_fine_grain_decoding",
      default=False,
      action="store_true")
    parser.add_argument(
      "--sql_use_cell_value",
      default=False,
      action="store_true")
    parser.add_argument(
      "--disconnect_deliberate_decoding",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--disconnect_deliberate_fine_grain_decoding",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--use_column_selection_loss",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--connect_to_sketch",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--add_noise",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--use_adamw",
      default=False,
      action="store_true"
    )


  @classmethod
  def add_model_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--rat_sql_lstm_hidden_size",
      type=int,
      default=150
    )
    parser.add_argument(
      "--rat_sql_hidden_size",
      type=int,
      default=300
    )
    parser.add_argument(
      "--rat_sql_action_embedding_dim",
      type=int,
      default=128
    )
    parser.add_argument(
      "--rat_sql_node_embedding_dim",
      type=int,
      default=64
    )
    parser.add_argument(
      "--rat_sql_schema_embedding_dim",
      type=int,
      default=300
    )
    parser.add_argument(
      "--use_coarse_to_fine",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--rat_sql_bert_embedding_size",
      type=int,
      default=768
    )



class SQLRerankingArgument:
  @classmethod
  def add_task_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--use_lr_scheduler",
      default=False,
      action="store_true"
    )
    parser.add_argument(
      "--use_adamw",
      default=False,
      action="store_true"
    )

  @classmethod
  def add_model_specific_args(cls, parser, root_dir):
    parser.add_argument(
      "--bert_hidden_size",
      type=int,
      default=768)