from relogic.tpukit.processors.xnli import XnliProcessor, MnliProcessor, MrpcProcessor, ColaProcessor

processors = {
  "cola": ColaProcessor,
  "mnli": MnliProcessor,
  "mrpc": MrpcProcessor,
  "xnli": XnliProcessor,
}