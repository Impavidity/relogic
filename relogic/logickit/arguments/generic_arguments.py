class GenericArguments:
  @classmethod
  def add_generic_arguments(cls, parser, root_dir):
    # Logging
    parser.add_argument(
      "--tb_writer_log_dir",
      default="./runs",
      type=str,
      help="The path for saving the tensorboad data"
    )

    # Dropout
    parser.add_argument(
      "--hidden_dropout_prob",
      default=0.1,
      type=float,
      help="The dropout for the output of the general encoder such as LSTM and BERT",
    )
