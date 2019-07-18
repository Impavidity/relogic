import relogic

relogic.download()

ner_model = relogic.Pipeline(component_names=["ner"])

example_doc = "诺瓦克·德约科维奇和西莫娜·哈勒普分別在温布尔登网球锦标赛贏得男子单打和女子单打冠军。"

annotation = ner_model(example_doc)