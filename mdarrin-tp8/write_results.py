import numpy as np
import pandas as pd

d = {'splex_host': {'test_accuracy': 0.6666666666666666, 'keras_testacc': 0.6, 'keras_testloss': 4.3069024085998535,
                    'keras_crossval': 0.435, 'sklean_crossval': 0.43499999999999994},
     'splex_micro': {'test_accuracy': 0.5333333333333333, 'keras_testacc': 0.4, 'keras_testloss': 7.686776638031006,
                     'keras_crossval': 0.67, 'sklean_crossval': 0.6950000000000001},
     'splex_env': {'test_accuracy': 0.4, 'keras_testacc': 0.6, 'keras_testloss': 5.455205917358398,
                   'keras_crossval': 0.35500002, 'sklean_crossval': 0.47000000000000003},
     'splex_full': {'test_accuracy': 0.5333333333333333, 'keras_testacc': 0.26666668,
                    'keras_testloss': 15.538191795349121, 'keras_crossval': 0.42, 'sklean_crossval': 0.515}}

df = pd.DataFrame.from_dict(d, orient="index")

df.to_csv("exports/results_splex.csv")