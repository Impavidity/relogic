import json
from types import SimpleNamespace

import relogic.utils.crash_on_ipy
from relogic.logickit.base.constants import MIXSENT_TASK
from relogic.logickit.dataflow import TASK_TO_DATAFLOW_CLASS_MAP, MixSentDataFlow
from relogic.logickit.tokenizer.tokenizer_roberta_xlm import RobertaXLMTokenizer

config = SimpleNamespace(
  **{
    "buckets": [(0, 100), (100, 250), (250, 512)],
    "max_seq_length": 512
  })

tokenizers = {
  "xlmr": RobertaXLMTokenizer.from_pretrained("xlmr.large.v0")
}

dataflow: MixSentDataFlow = TASK_TO_DATAFLOW_CLASS_MAP[MIXSENT_TASK](
  task_name=MIXSENT_TASK,
  config=config,
  tokenizers=tokenizers,
  label_mapping=None)

examples = [
  {"text_a": ["EFE", "-", "Cantabria", "Madrid", ",", "23", "may", "(", "EFE", ")", "."],
   "text_b": ["De", "este", "modo", ",", "podr\u00edan", "ser", "unos", "1.500", "los", "milicianos",
              "que", "se", "han", "rendido", "en", "los", "tres", "\u00faltimos", "d\u00edas", ",",
              "mientras", "se", "supone", "que", "los", "mil", "componentes", "restantes", "del", "ESL",
              "tratan", "de", "llegar", "a", "Israel", "."],
   "text_c": ["-", "Cantabria", "Madrid", ",", "23", "d\u00edas", ",", "mientras", "se", "supone", "que",
              "los", "mil", "componentes", "restantes", "del", "ESL", "tratan", "de", "llegar", "a", "Israel"],
   "span_a": [1, 6], "span_b": [18, 35], "span_c_a": [0, 5], "span_c_b": [5, 22]},
{"text_a": ["El", "entrenador", "de", "la", "Real", "Sociedad", "Javier", "Clemente", "dijo", "hoy", "que", "el",
            "club", "guipuzcoano", "no", "tiene", "opciones", "de", "acceder", "al", "mercado", "espa\u00f1ol",
            ",", "ya", "que", "a", "su", "juicio", "\"", "es", "imposible", "\"", "fichar", "jugadores", "de", "\"",
            "cierto", "nivel", "\"", ",", "por", "lo", "que", "afirm\u00f3", "que", "esa", "idea", "\"", "hay", "que",
            "quit\u00e1rsela", "de", "la", "cabeza", "\"", "."],
 "text_b": ["Este", "portavoz", "sindical", "dijo", "tras", "la", "desconvocatoria", "de", "la", "huelga", "que",
            "este", "preacuerdo", "\"", "puede", "ser", "perjudicial", "y", "regresivo", "para", "las", "condiciones",
            "laborales", "\"", "de", "los", "trabajadores", ",", "si", "bien", "el", "sindicato", "respet\u00f3", "su",
            "decisi\u00f3n", "."],
 "text_c": ["club", "guipuzcoano", "no", "tiene", "opciones", "de", "acceder", "al", "mercado", "espa\u00f1ol", ",",
            "ya", "que", "a", "su", "juicio", "\"", "es", "imposible", "\"", "fichar", "jugadores", "de", "\"",
            "cierto", "nivel", "\"", "sindical", "dijo", "tras", "la", "desconvocatoria", "de", "la", "huelga",
            "que", "este", "preacuerdo", "\"", "puede", "ser", "perjudicial", "y", "regresivo", "para"],
 "span_a": [12, 39], "span_b": [2, 20], "span_c_a": [0, 27], "span_c_b": [27, 45]},
{"text_a": ["Todos", "estos", "temas", "se", "tratar\u00e1n", "en", "la", "Comisi\u00f3n", "Mixta", "RENFE-Junta",
            "de", "Comunidades", ",", "creada", "para", "negociar", "todos", "los", "problemas", "que", "existen",
            "al", "respecto", "en", "la", "Comunidad", "Aut\u00f3noma", "y", "de", "la", "cual", "formar\u00e1",
            "parte", "el", "Ayuntamiento", "de", "Alc\u00e1zar", "como", "observador", "."],
 "text_b": ["Este", "reconocido", "profesional", ",", "con", "estudios", "en", "Nueva", "York", ",", "Tokio",
            "y", "Buenos", "Aires", ",", "tiene", "una", "amplia", "experiencia", "en", "el", "dise\u00f1o",
            "de", "espacios", "multidisciplinares", "y", "con", "fines", "culturales", "y", "entre", "otros",
            "proyectos", "trabaja", "en", "la", "actualidad", "en", "el", "dise\u00f1o", "de", "un", "centro",
            "cultural", "para", "Filadelfia", ",", "que", "albergar\u00e1", "a", "la", "orquesta", "titular",
            "de", "esa", "ciudad", "."],
 "text_c": ["al", "respecto", "en", "la", "Comunidad", "Aut\u00f3noma", "y", "de", "la", "cual", "formar\u00e1",
            "parte", "el", "Ayuntamiento", "de", "Alc\u00e1zar", "como", "observador", ".", "experiencia", "en",
            "el", "dise\u00f1o", "de", "espacios", "multidisciplinares", "y", "con", "fines", "culturales", "y",
            "entre", "otros", "proyectos", "trabaja", "en", "la", "actualidad", "en", "el", "dise\u00f1o", "de",
            "un", "centro", "cultural", "para", "Filadelfia"],
 "span_a": [21, 40], "span_b": [18, 46], "span_c_a": [0, 19], "span_c_b": [19, 47]},
{"text_a": ["Posibilidad", "de", "alg\u00fan", "banco", "de", "niebla", "en", "el", "sureste", "."],
 "text_b": ["Temperaturas", "en", "ascenso", "ligero", "en", "el", "\u00e1rea", "del", "cant\u00e1brico",
            "oriental", ",", "La", "Rioja", ",", "Navarra", ",", "Arag\u00f3n", "y", "en", "Canarias", "y",
            "sin", "cambios", "en", "el", "resto", "."],
 "text_c": ["de", "niebla", "en", "el", "sureste", "en", "ascenso", "ligero", "en", "el", "\u00e1rea", "del",
            "cant\u00e1brico", "oriental", ",", "La", "Rioja"],
 "span_a": [4, 9], "span_b": [1, 13], "span_c_a": [0, 5], "span_c_b": [5, 17]}]

dataflow.update_with_jsons(examples)

def check_example(example):
  a_tokens = example.a_tokens
  b_tokens = example.b_tokens
  c_tokens = example.c_tokens
  span_a_selected_index = example.span_a_selected_index
  span_b_selected_index = example.span_b_selected_index
  span_c_a_selected_index = example.span_c_a_selected_index
  span_c_b_selected_index = example.span_c_b_selected_index
  assert((len(span_a_selected_index) + len(span_b_selected_index)) ==
         (len(span_c_a_selected_index) + len(span_c_b_selected_index)))
  assert(a_tokens[span_a_selected_index[0]] == c_tokens[span_c_a_selected_index[0]])
  assert (b_tokens[span_b_selected_index[0]] == c_tokens[span_c_b_selected_index[0]])

for mb in dataflow.get_minibatches(minibatch_size=2):
  mb.generate_input("cpu", True)
  for example in mb.examples:
    check_example(example)

print("Passed.")

