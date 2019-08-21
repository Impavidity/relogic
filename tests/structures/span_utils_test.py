from relogic.structures import enumerate_spans

sentence = ["My", "Country", "and", "Me"]

print(enumerate_spans(sentence=sentence, max_span_width=3))
