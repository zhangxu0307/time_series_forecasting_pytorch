import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from code.util import *

#ts, data = load_data("../data/NSW2013.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
# ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
#s, data = load_data("../data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
#ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

dta=pd.Series(data[200:500])
fig = plt.figure(figsize=(12, 8))
ax1= fig.add_subplot(121)
ax2 = fig.add_subplot(122)
diff1 = dta.diff(1)
dta.plot(ax=ax1)
diff1.plot(ax=ax2)

plt.show()