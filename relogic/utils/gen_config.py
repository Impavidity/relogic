from bullet import Bullet, Check, YesNo, Input
from bullet import utils, colors
from bullet.client import myInput
import json
import os
import re

"""
{
  "config_sections": ["key1"],
  "key1" : {
      "prompt": "The description",
      "prompt_type": "Bullet",
      "choices": [],
      "branching": {
        "answer1": "key1.answer1"
      } 
  }
}
"""

def check_configuration(configuration):
  return True

class ListInput:
  def __init__(
        self,
        prompt,
        default="",
        indent=0,
        word_color=colors.foreground["default"],
        strip=True,
        pattern="",
        separator=","
  ):
    self.indent = indent
    if not prompt:
      raise ValueError("Prompt can not be empty!")
    self.default = "[{}]".format(default) if default else ""
    self.prompt = prompt
    self.word_color = word_color
    self.strip = strip
    self.pattern = pattern
    self.separator = separator

  def valid(self, ans):
    if not bool(re.match(self.pattern, ans)):
      utils.moveCursorUp(1)
      utils.forceWrite(' ' * self.indent + self.prompt + self.default)
      utils.forceWrite(' ' * len(ans))
      utils.forceWrite('\b' * len(ans))
      return False
    return True

  def launch(self):
    utils.forceWrite(' ' * self.indent + self.prompt + self.default)
    sess = myInput(word_color=self.word_color)
    if not self.pattern:
      result = sess.input()
    else:
      while True:
        result = sess.input()
        if self.valid(result):
          break
    if self.strip:
      result = result.strip()
    if result:
      return result.split(self.separator)
    else:
      return []

class ConfiguredPrompt:
  def __init__(self, configuration):
    self.configuration = configuration
    self.results = {}

  def rlaunch(self, key, depth):
    results = {}
    section_config = self.configuration[key]
    if section_config["prompt_type"] == "Check":
      ui = Check(section_config["prompt"],
                 choices=section_config["choices"],
                 check=" √",
                 margin=2,
                 check_color=colors.bright(colors.foreground["red"]),
                 check_on_switch=colors.bright(colors.foreground["red"]),
                 background_color=colors.background["black"],
                 background_on_switch=colors.background["white"],
                 word_color=colors.foreground["white"],
                 word_on_switch=colors.foreground["black"],
                 indent=depth * 2)
      choices = ui.launch()
      branching = section_config.get("branching")
      if branching is not None:
        for sub_key in choices:
          branching_key = branching.get(sub_key)
          if branching_key is not None:
            if branching_key.startswith("."):
              results[sub_key] = self.rlaunch("{}{}".format(key, branching_key), depth)
            else:
              results[sub_key] = self.rlaunch(branching_key, depth)
          else:
            raise ValueError("the key {} is not in branching {}".format(sub_key, branching.keys()))
        return results
      else:
        return results
    if section_config["prompt_type"] == "ListInput":
      ui = ListInput(section_config["prompt"],
        word_color=colors.foreground["yellow"],
        indent=depth * 2)
      results = ui.launch()
      return results
    if section_config["prompt_type"] == "Input":
      ui = Input(section_config["prompt"],
        word_color=colors.foreground["yellow"],
        indent=depth * 2)
      results = ui.launch()
      return results
    if section_config["prompt_type"] == "YesNo":
      ui = YesNo(section_config["prompt"],
        word_color=colors.foreground["yellow"],
        default=section_config["default"] if "default" in section_config else 'y',
        indent=depth * 2)
      results = ui.launch()
      return results
    if section_config["prompt_type"] == "Bullet":
      ui = Bullet(section_config["prompt"],
            choices=section_config["choices"],
            bullet=" >",
            margin=2,
            bullet_color=colors.bright(colors.foreground["cyan"]),
            background_color=colors.background["black"],
            background_on_switch=colors.background["black"],
            word_color=colors.foreground["white"],
            word_on_switch=colors.foreground["white"],
            indent=depth * 2)
      results = ui.launch()
      return results
    if section_config["prompt_type"] == "GoTo":
      for sub_key in section_config["goto"]:
        if sub_key.startswith("."):
          sub_value = self.rlaunch("{}{}".format(key, sub_key), depth)
          sub_key = sub_key[1:]
        else:
          sub_value = self.rlaunch(sub_key, depth)
        if isinstance(sub_value, bool) or sub_value:
          # If True/False or other non-empty data (! "", [], {})
          results[sub_key] = sub_value
      return results

  def launch(self):
    for section in self.configuration["config_sections"]:
      utils.cprint("We are now configuring {} section".format(section), color=colors.foreground["default"])
      self.results[section] = self.rlaunch(section, 1)
    utils.cprint("You finish all the configurations, here is your full configurations",
                 color=colors.foreground["default"])
    utils.cprint(json.dumps(self.results, indent=2), color=colors.foreground["default"])
    while True:
      ui = Input("Where are you going to save the configuration? ",
          word_color=colors.foreground["yellow"],
          default="configurations/default.json")
      path = ui.launch()
      if path:
        path_dir = os.path.dirname(path)
        if os.path.exists(path_dir):
          with open(path, 'w') as fout:
            fout.write(json.dumps(self.results, indent=2))
          break
        else:
          utils.cprint("The dir `{}` does not exist".format(path_dir))
      else:
        utils.cprint("Invalid input")
    utils.cprint("Done!")





  @classmethod
  def from_config(cls, config_file):
    with open(config_file) as fin:
      configuration = json.load(fin)
      if check_configuration(configuration):
        return cls(configuration=configuration)
      else:
        raise ValueError("Configuration Error")



file_dir = os.path.dirname(os.path.abspath(__file__))
cls = ConfiguredPrompt.from_config(os.path.join(file_dir, "..", "..", "configurations/full_configuration.json"))
cls.launch()
