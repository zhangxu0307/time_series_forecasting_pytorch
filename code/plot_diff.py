import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from code.util import *

#ts, data = load_data("../data/NSW2013.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
# ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
#s, data = load_data("../data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
#ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")


data = data[300:500]
plt.plot(data)
plt.savefig("../data/original.eps", format="eps")

diffData = np.diff(data)
plt.plot(diffData)
plt.savefig("../data/diff_data.eps", format="eps")


