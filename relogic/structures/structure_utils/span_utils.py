from typing import Callable, List, Tuple, TypeVar
from relogic.structures.token import Token

T = TypeVar("T", str, Token)

def enumerate_spans(sentence: List[T],
                    offset: int = 0,
                    max_span_width: int = None,
                    min_span_width: int = 1,
                    filter_function: Callable[[List[T]], bool] = None) -> List[Tuple[int, int]]:
  """
  Span is exclusive

  Args:
    sentence (List[T])
    offset (int)
    max_span_width (int)
    min_span_width (int)
    filter_function (Callable):
  """
  max_span_width = max_span_width or len(sentence)
  filter_function = filter_function or (lambda x: True)
  spans: List[Tuple[int, int]] = []

  for start_index in range(len(sentence)):
    last_end_index = min(start_index + max_span_width, len(sentence))
    first_end_index = min(start_index + min_span_width, len(sentence))
    for end_index in range(first_end_index, last_end_index + 1):
      # add 1 to end_index range because span indices are exclusive
      start = offset + start_index
      end = offset + end_index
      if filter_function(sentence[slice(start_index, end_index)]):
        spans.append((start, end))
  return spans


