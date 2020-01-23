READING_COMPREHENSION_TASKS = ["squad11", "squad20"]
AUTO_SPAN = ["auto_span_srl"]
NEVER_SPLIT = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]",
              '[ORGANIZATION-SUBJ]', '[PERSON-OBJ]', '[PERSON-SUBJ]',
              '[ORGANIZATION-OBJ]', '[NUMBER-OBJ]', '[DATE-OBJ]',
              '[NATIONALITY-OBJ]', '[LOCATION-OBJ]', '[TITLE-OBJ]',
              '[CITY-OBJ]', '[MISC-OBJ]', '[COUNTRY-OBJ]',
              '[CRIMINAL_CHARGE-OBJ]', '[RELIGION-OBJ]',
              '[DURATION-OBJ]', '[URL-OBJ]', '[STATE_OR_PROVINCE-OBJ]',
              '[IDEOLOGY-OBJ]', '[CAUSE_OF_DEATH-OBJ]')

JOINT_SRL_TASK = "joint_srl"
PIPE_SRL_TASK = "pipe_srl"
SRL_LABEL_SPAN_BASED = "srl_label_span_based"
SRL_LABEL_SEQ_BASED = "srl_label_seq_based"
SPAN_REPR_KENTON_LEE = "kenton_lee"
SPAN_REPR_AVE_MAX = "ave_max"
SPAN_REPR_AVE = "ave"
SRL_PREDICATE_EXTRA_SURFACE = "srl_predicate_extra_surface"
PREDICATE_DETECTION_TASK = "predicate_detection"


ECP_TASK = "ecp"


# Language Code based on ISO 639-1 codes
LANG_EN = "en"
LANG_zH = "zh"

POINTWISE_TASK = "pointwise"
PAIRWISE_TASK = "pairwise"

IR_TASK = "ir"
DOCIR_TASK = "doc_ir"
IR_SIAMESE_TASK = "ir_siamese"
IR_SAMCNN_TASK = "ir_samcnn"


NER_TASK = "ner"
POS_TASK = "pos"
ENTITY_TYPE_CLASSIFICATION = "entity_type_classification"
SEQUENCE_LABELING_TASK = "sequence_labeling"
DEP_PARSING_TASK = "dep_parsing"
SEQUENCE_CLASSIFICATION_TASK = "sequence_classification"
PARALLEL_MAPPING_TASK = "parallel_mapping"
PARALLEL_TEACHER_STUDENT_TASK = "parallel_teacher_student"
MIXSENT_TASK = "mixsent"

LANGUAGE_IDENTIFICATION_IR = "language_identification_ir"
LANGUAGE_IDENTIFICATION_SEQ = "language_identification_seq"


SINGLETON = []
SIAMESE = [IR_SIAMESE_TASK, IR_SAMCNN_TASK]
TRIPLET = [PAIRWISE_TASK]

DISTILL_TASKS = ["parallel_teacher_student", "mixsent"]

ENCODING_TASK = "encoding"

AUXILIARY_TRAINING = "auxiliary_training"
ITERATIVE_TRAINING = "iterative_training"
ADVERSARIAL_TRAINING = "adversarial_training"

TRAIN_DISCRIMINATOR = "train_discriminator"
TRAIN_GENERATOR = "train_generator"

SKIP_LOSS_TASK = [LANGUAGE_IDENTIFICATION_IR, LANGUAGE_IDENTIFICATION_SEQ]
SKIP_EVAL_TASK = [LANGUAGE_IDENTIFICATION_IR, LANGUAGE_IDENTIFICATION_SEQ]


