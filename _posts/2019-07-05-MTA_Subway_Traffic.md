---
layout: post
title: An Analysis of MTA Subway Traffic
---

While most generally prefer to avoid the hordes of pedestrians that are so characteristic of New York City's large subway stations, for this project, my aim was to find them. Using publicly-available data provided by the New York City Metropolitan Transportation Authority (MTA) on foot traffic at subway station turnstiles (data available [here](http://web.mta.info/developers/turnstile.html)), I set out to assist a (hypothetical) client optimize placement of street teams to reach the most people possible. The client hosts an annual gala in NYC for which they commonly attract visitors by enlisting street teams to collect their e-mail addresses at busy pedestrian intersections. Presumably the more people they can interact with on a daily basis, the more e-mail addresses they can collect to follow-up with later on and persuade potential attendees to come to the gala. Thankfully, the MTA turnstile data tracks the cumulative counts of pedestrians coming and going from subway stations and reports the aggregate counts every 4 hours, providing a useful metric of foot traffic density at every subway station, date, and time of day. This seemed like a good place to start to provide recommendations to the client regarding where and when they should place their street teams. 

I started by loading in the data, which is available in weekly .csv files. The following script was used to allow for flexibility in how much data was loaded in at a time:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import time
from datetime import datetime
from datetime import timedelta

frames = []
start = datetime(2019,6,22)
count = 1
while count <= 8:
    df = pd.read_csv("http://web.mta.info/developers/data/nyct/turnstile/turnstile_"+
                    ((start-timedelta(days=7)).strftime("%y%m%d"))+".txt")
    frames.append(df)
    start = (start-timedelta(days=7))
    count += 1
    print(len(frames))
df = pd.concat(frames)
```

Cleaning the data proved rather difficult, since there were several instances of duplicate rows (due to 'recovery' entries), counter restarts, and non-uniform time intervals, and because I was primarily interested in daily totals rather than counts every 4 hours. I also wanted to extract the differences in counts between days (i.e. the number of people who entered or exited that day) rather than the running totals. My primary steps to get the data cleaned and in that format were:

```
mta.columns = mta.columns.str.strip()
mta['date_time'] = mta['DATE'] + ' ' + mta['TIME']
mta['date_time'] = mta['date_time'].apply(parser.parse)

keys = ['C/A', 'UNIT', 'SCP', 'STATION', 'date_time']
mta.drop_duplicates(subset=keys, inplace=True)

mta_by_date = mta.groupby(['C/A', 'UNIT', 'SCP', 'STATION', 'DATE'], 
                          as_index = False)['ENTRIES', 'EXITS'].first()

mta_by_date[["PREV_DATE", "PREV_ENTRIES", "PREV_EXITS"]] = \
    mta_by_date.groupby(["C/A", "UNIT", "SCP", "STATION"])["DATE", "ENTRIES", "EXITS"].\
    transform(lambda x: x.shift(1))

mta_by_date.dropna(subset=["PREV_DATE"], axis=0, inplace=True)
mta_by_date = mta_by_date[mta_by_date["ENTRIES"] >= mta_by_date["PREV_ENTRIES"]]
mta_by_date = mta_by_date[mta_by_date["EXITS"] >= mta_by_date["PREV_EXITS"]]
mta_by_date.drop(columns='DATE', axis=1, inplace = True)
mta_by_date.rename(index = str, columns = {'PREV_DATE': 'DATE'}, inplace = True)
mta_by_date['DATE'] = pd.to_datetime(mta_by_date.DATE, format = '%m/%d/%Y')

dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
mta_by_date['weekday'] = mta_by_date.DATE.apply(datetime.weekday)
mta_by_date['weekday'] = mta_by_date['weekday'].map(dayOfWeek)

mta_by_date['daily_entries'] = mta_by_date['ENTRIES'] - mta_by_date['PREV_ENTRIES']
mta_by_date['daily_exits'] = mta_by_date['EXITS'] - mta_by_date['PREV_EXITS']

mta_by_date = mta_by_date[(mta_by_date.daily_entries < 20000) & (mta_by_date.daily_exits < 20000)]
```

With the data cleaned and prepared, I was ready to do some exploratory analysis. 

Some of the more predictable aspects of MTA pedestrian traffic quickly became clear. Weekdays were by far busier than weekends, with peak traffic generally occurring around the middle of the week. The following plots summarize this trend well:

![weekday_boxplot](https://github.com/peterilhardt/peterilhardt.github.io/tree/master/images/Weekday_Boxplot.png)

![daily_traffic_stations](https://github.com/peterilhardt/peterilhardt.github.io/tree/master/images/Daily_Traffic_Stations.png)

It was also clear that some stations were far busier than others. The distribution of total foot traffic by station was highly right-skewed, shown here on a log scale:

![hist_log_total_traffic](https://github.com/peterilhardt/peterilhardt.github.io/tree/master/images/Hist_log_Total_Traffic.png)

The busiest stations themselves were somewhat predictable, capped by Penn Station, Grand Central Station, Herald Square, Union Square, and Times Square:

![bar_busiest_stations](https://github.com/peterilhardt/peterilhardt.github.io/tree/master/images/Bar_Busiest_Stations_2.png)




To be continued... 







