import tensorflow as tf
import os
import time

from relogic.tpukit.model.classifier_model import model_fn_builder
from relogic.tpukit.inference import modeling
from relogic.tpukit.processors import processors
from relogic.tpukit.input_fn import file_based_input_fn_builder, file_based_convert_examples_to_features, PaddingInputExample
from relogic.tpukit.tokenizer import tokenization

class Trainer(object):
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS
    self.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    self.task_name = FLAGS.task_name.lower()
    self.tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    processor = processors[self.task_name]()
    self.label_list = processor.get_labels()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

    if not FLAGS.do_pretraining and FLAGS.do_train:

      # Setup Training
      self.train_examples = None
      num_train_steps = None
      num_warmup_steps = None
      if FLAGS.do_train:
        self.train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
          len(self.train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        train_file = os.path.join(self.FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
          self.train_examples, self.label_list, self.FLAGS.max_seq_length, self.tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        self.train_input_fn = file_based_input_fn_builder(
          input_file=train_file,
          seq_length=self.FLAGS.max_seq_length,
          is_training=True,
          drop_remainder=True)

        # Setup Validation
        self.dev_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(self.dev_examples)
        if FLAGS.use_tpu:
          # TPU requires a fixed batch size for all batches, therefore the number
          # of examples must be a multiple of the batch size, or else examples
          # will get dropped. So we pad with fake examples which are ignored
          # later on. These do NOT count towards the metric (all tf.metrics
          # support a per-instance weight, and these get a weight of 0.0).
          while len(self.dev_examples) % FLAGS.eval_batch_size != 0:
            self.dev_examples.append(PaddingInputExample())
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
          self.dev_examples, self.label_list, FLAGS.max_seq_length, self.tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(self.dev_examples), num_actual_eval_examples,
                        len(self.dev_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        self.eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
          assert len(self.dev_examples) % FLAGS.eval_batch_size == 0
          self.eval_steps = int(len(self.dev_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        self.eval_input_fn = file_based_input_fn_builder(
          input_file=eval_file,
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=eval_drop_remainder)

    if FLAGS.do_pretraining:
      num_train_steps = FLAGS.num_train_steps
      num_warmup_steps = FLAGS.num_warmup_steps

    self.num_train_steps = num_train_steps

    model_fn = model_fn_builder(
      bert_config=self.bert_config,
      num_labels=len(self.label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

    self.estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  def train(self):
    # heading = lambda s: utils.heading(s, '(' + self.config.model_name + ')')
    current_step = 0

    if not self.FLAGS.do_pretraining:

      start_timestamp = time.time()
      # while current_step < self.num_train_steps:
      #   next_checkpoint = min(current_step + self.FLAGS.eval_dev_every, self.num_train_steps)
      self.estimator.train(input_fn=self.train_input_fn, max_steps=self.num_train_steps)
      # current_step = next_checkpoint
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      self.num_train_steps, int(time.time() - start_timestamp))
      tf.logging.info('Starting to evaluate at step %d', self.num_train_steps)

      self.evaluate_task()

    else:
      raise NotImplementedError()

  def evaluate_task(self):


    result = self.estimator.evaluate(input_fn=self.eval_input_fn, steps=self.eval_steps)

    output_eval_file = os.path.join(self.FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
