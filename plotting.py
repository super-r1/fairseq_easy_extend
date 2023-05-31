import matplotlib.pyplot as plt
import numpy as np
import re
import os


# Still to do@
path = "/content/checkpoint/{}/log_file.log"
models = ["bleu", "bleurt","sacrebleu", "comet", "meteor", "chrf"]
regex = r"loss=([\d.]+), reward=([\d.]+),.*num_updates=([\d]+).*lr=([\d.e-]+).*"
for model in models: 
  text_path = path.format(model)
  if os.path.exsits(text_path):
    with open(text_path, 'r') as file:
       for line in file:
          z = re.match(line.strip())
          if z:
            z.group(1)
            z.group(2)
            z.group(3)
            z.group(4)
# Open the text file and get the loss, number of steps and reward and learning rate
