from relogic.pipelines.core import Pipeline
import relogic.utils.crash_on_ipy

pipeline = Pipeline(
  component_names=["predicate_detection"],
  component_model_names= {"predicate_detection": "spacy"})

from relogic.structures.sentence import Sentence

sent1 = Sentence(
  text="Barack Obama can't go to Paris.")
sent2 = Sentence(
  text="Pulmonary alveolar proteinosis (PAP) is a rare lung disorder characterized by an abnormal accumulation of surfactant-derived lipoprotein compounds within the alveoli of the lung."
)
sent3 = Sentence(text="""Munthe was also a member, board member and chairman of a number of associations, companies, foundations, guilds and societies; board member of the Swedish Exhibition and Congress Centre 1928-1951, of the Stockholm Exhibition in 1930, of "Svenska slöjdföreningen" (Swedish Sloyd Association) 1931-1943 and 1944-1946 (chairman of the union section), of the Sweden–America Foundation\'s Zorn Scholarship 1929-1930, of the Gothenburg City Theatre 1935-1945, of the AB Gbgssystemet 1938-1945, of the "Svenska hemslöjdsföreningens riksförbund" ("Swedish National Home Sloyd Association") 1945-1950 and of the Drottningholm Theatre Museum from 1945.; member of the "Föreningen Svensk hemslöjd" ("Swedish Home Sloyd Association") 1945-1947, of the "Svenska orientsällskapet" ("Swedish Orient Society") from 1950, of the "Svensk-italienska sällskapet" ("Swedish-Italian Society") from 1952, of the "Konsthantverkarnas gille" ("Artisans\' Guild") 1952-1958, of the Foundation Natur & Kultur 1952, of the "Statens kommitté för social upplysning" ("State Committee for Social Enlightenment") in 1946 and of the Tourist Investigation at the Ministry of Trade in 1948; chairman of the Humanist Department of Stockholm University College Student Union from 1918-1920, of the Stockholm Federation of Student Unions in 1924, of the "Föreningen Göteborgs konsthantverkare" ("Association of Gothenburg\'s Artisans") 1929-1945, of the Stockholm University College Student Union 1923-1924, of the "Svensk-Jugoslaviska Sällskapet i Göteborg" ("Swedish-Yugoslav Society in Gothenburg") from 1939, of the Swedish Tourist Association\'s Gothenburg Board 1938-1943 and of the "Föreningen Konstfliten-Bohusslöjd" ("Handicraft-Bohuslän Sloyd Association") 1940-1944.; vice chairman of the "Svensk-Pakistanska vänskapsföreningen" ("Swedish-Pakistani Friendship Association") from 1953.""")

pipeline.execute([sent1, sent2, sent3])
print(sent1)
print(sent2)
print(sent3)

pipeline = Pipeline(
  component_names=["predicate_detection"],
  component_model_names={"predicate_detection": "pd-conll12"},
  )
sent1 = Sentence(
  text="Barack Obama can't go to Paris.")
sent2 = Sentence(
  text="Pulmonary alveolar proteinosis (PAP) is a rare lung disorder characterized by an abnormal accumulation of surfactant-derived lipoprotein compounds within the alveoli of the lung."
)
sent3 = Sentence(text="""Munthe was also a member, board member and chairman of a number of associations, companies, foundations, guilds and societies; board member of the Swedish Exhibition and Congress Centre 1928-1951, of the Stockholm Exhibition in 1930, of "Svenska slöjdföreningen" (Swedish Sloyd Association) 1931-1943 and 1944-1946 (chairman of the union section), of the Sweden–America Foundation\'s Zorn Scholarship 1929-1930, of the Gothenburg City Theatre 1935-1945, of the AB Gbgssystemet 1938-1945, of the "Svenska hemslöjdsföreningens riksförbund" ("Swedish National Home Sloyd Association") 1945-1950 and of the Drottningholm Theatre Museum from 1945.; member of the "Föreningen Svensk hemslöjd" ("Swedish Home Sloyd Association") 1945-1947, of the "Svenska orientsällskapet" ("Swedish Orient Society") from 1950, of the "Svensk-italienska sällskapet" ("Swedish-Italian Society") from 1952, of the "Konsthantverkarnas gille" ("Artisans\' Guild") 1952-1958, of the Foundation Natur & Kultur 1952, of the "Statens kommitté för social upplysning" ("State Committee for Social Enlightenment") in 1946 and of the Tourist Investigation at the Ministry of Trade in 1948; chairman of the Humanist Department of Stockholm University College Student Union from 1918-1920, of the Stockholm Federation of Student Unions in 1924, of the "Föreningen Göteborgs konsthantverkare" ("Association of Gothenburg\'s Artisans") 1929-1945, of the Stockholm University College Student Union 1923-1924, of the "Svensk-Jugoslaviska Sällskapet i Göteborg" ("Swedish-Yugoslav Society in Gothenburg") from 1939, of the Swedish Tourist Association\'s Gothenburg Board 1938-1943 and of the "Föreningen Konstfliten-Bohusslöjd" ("Handicraft-Bohuslän Sloyd Association") 1940-1944.; vice chairman of the "Svensk-Pakistanska vänskapsföreningen" ("Swedish-Pakistani Friendship Association") from 1953.""")

pipeline.execute([sent1, sent2, sent3])

print(sent1)
print(sent2)
print(sent3)