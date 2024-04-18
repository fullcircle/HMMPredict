import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt
# https://pyquantnews.com/use-markov-models-to-detect-regime-changes/


data = yf.download("NVDA")
returns = np.log(data.Close / data.Close.shift(1))
range = (data.High - data.Low)
features = pd.concat([returns, range], axis=1).dropna()
features.columns = ["returns", "range"]

model = hmm.GaussianHMM(
  n_components=3,
  covariance_type="full",
  n_iter=1000,
)
model.fit(features)
states = pd.Series(model.predict(features), index=data.index[1:])
states.name = "state"
states.hist()

color_map = {
    0.0: "green",
    1.0: "orange",
    2.0: "red"
}
(
    pd.concat([data.Close, states], axis=1)
    .dropna()
    .set_index("state", append=True)
    .Close
    .unstack("state")
    .plot(color=color_map, figsize=[16, 12])
)
# Show the plot
plt.show()