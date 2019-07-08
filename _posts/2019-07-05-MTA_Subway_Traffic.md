---
layout: post
title: An Analysis of New York City MTA Subway Traffic
---

While most generally prefer to avoid the hordes of pedestrians that are so characteristic of New York City's large subway stations, for this project, my aim was to find them. Using publicly-available data provided by the New York City Metropolitan Transportation Authority (MTA) on foot traffic at subway station turnstiles (data available [here](http://web.mta.info/developers/turnstile.html)), I set out to assist a (hypothetical) client optimize placement of street teams to reach the most people possible. The client hosts an annual gala in NYC for which they commonly attract visitors by enlisting street teams to collect their e-mail addresses at busy pedestrian intersections. Presumably the more people they can interact with on a daily basis, the more e-mail addresses they can collect to follow-up with later on and persuade potential attendees to come to the gala. Thankfully, the MTA turnstile data tracks the cumulative counts of pedestrians coming and going from subway stations and reports the aggregate counts every 4 hours, providing a useful metric of foot traffic density at every subway station, date, and time of day. This seemed like a good place to start to provide recommendations to the client regarding where and when they should place their street teams. 

I started by loading in the data, which is available in weekly .csv files. The following script was used to allow for flexibility in how much data was loaded in at a time:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

![weekday_boxplot]({{ site.url }}/images/Weekday_Boxplot.png)

![daily_traffic_stations]({{ site.url }}/images/Daily_Traffic_Stations.png)

It was also clear that some stations were far busier than others. In fact, it seemed the bulk of the foot traffic in NYC was largely concentrated at approximately 10 stations. The distribution of total traffic by station was thus highly right-skewed, shown here on a log scale:

![hist_log_total_traffic]({{ site.url }}/images/Hist_log_Total_Traffic.png)

The busiest stations themselves were not particularly surprising, topped by some of the more well-known NYC stations: Penn Station, Grand Central Station, Herald Square, Union Square, and Times Square:

![bar_busiest_stations]({{ site.url }}/images/Bar_Busiest_Stations_2.png)

To produce a list of these busiest stations, I used the following procedure:

```
station_total = by_day.groupby('STATION')['total_traffic']\
	.sum().reset_index().sort_values('total_traffic', ascending=False)
top_stations = station_total.iloc[:9,:].STATION.tolist()
```

Next, I wanted to discern how time of day factored into the traffic picture. Not surprisingly, traffic appeared to peak around midday and tail off at night, but interestingly, peak entry traffic (people going into the station) was somewhat offset from peak exit traffic. Exits generally appeared to peak around noon, whereas entries peaked somewhat later (end of the business day). While it would likely be irrelevant to a street team whether a passerby is entering or exiting the station, it *would* likely affect whether that person would stop to talk to a marketer. The two plots below show the typical traffic trends for select busy stations over the course of a day as well as the corresponding entry and exit trends for a day at Penn Station:

![traffic_by_time]({{ site.url }}/images/Traffic_by_Time.png)

![penn_station_entry_exit](peterilhardt.github.io/images/Penn_Station_Entry_Exit.png)

This entry-exit disparity also manifested in bivariate analyses of the two variables. Shown below are scatterplots of exit vs. entry traffic summed over each day and then in raw form (reported every 4 hours). It is clear that in looking at each day in the aggregate, entry traffic is highly correlated with exit traffic since stations that are busy in the morning will likely also be busy in the evening. When broken down into time intervals, however, we see different slopes corresponding to different times of day. This is further evidence of an entry-exit lag in daily foot traffic. 

![entry_vs_exit_scatter](peterilhardt.github.io/images/Enter_vs_Exit_Scatter.png)

![entry_vs_exit_time_scatter](peterilhardt.github.io/images/Enter_vs_Exit_Time_Scatter.png)

Finally, I wanted to incorporate location data to visualize peak foot traffic in geographical space. For this I used the *folium* module and imported station location data from the MTA public repository (data [here](http://web.mta.info/developers/data/nyct/subway/Stations.csv)). This file has the latitude and longitude coordinates for each station. I imported the data, cleaned it (including renaming some of the busiest stations due to name discrepancies between files), and merged it with the turnstile files using the following script:

```
loc_url = 'http://web.mta.info/developers/data/nyct/subway/Stations.csv'
mta_loc = pd.read_csv(loc_url)

mta_loc.rename(columns={'Stop Name': 'STATION', 'GTFS Latitude': 'LAT', 'GTFS Longitude': 'LONG'}, inplace = True)
mta_loc['STATION'] = mta_loc.STATION.str.upper()
mta_loc['STATION'] = mta_loc.STATION.str.replace(' - ', '-')
mta_loc = mta_loc.drop_duplicates('STATION').drop(columns = ['Station ID', 'Complex ID', 'GTFS Stop ID',
                                                            'Division', 'Structure', 'North Direction Label',
                                                            'South Direction Label'])

mta_loc.STATION = mta_loc.STATION.replace('34 ST-PENN STATION', '34 ST-PENN STA')
mta_loc.STATION = mta_loc.STATION.replace('GRAND CENTRAL-42 ST', 'GRD CNTRL-42 ST')
mta_loc.STATION = mta_loc.STATION.replace('42 ST-PORT AUTHORITY BUS TERMINAL', '42 ST-PORT AUTH')
mta_loc.STATION = mta_loc.STATION.replace('59 ST-COLUMBUS CIRCLE', '59 ST COLUMBUS')
mta_loc.STATION = mta_loc.STATION.replace('47-50 STS-ROCKEFELLER CTR', '47-50 STS ROCK')
mta_loc.STATION = mta_loc.STATION.replace('FLUSHING-MAIN ST', 'FLUSHING-MAIN')
mta_loc.STATION = mta_loc.STATION.replace('JACKSON HTS-ROOSEVELT AV', 'JKSN HT-ROOSVLT')

mta_join = by_day.merge(mta_loc, how = 'inner', on = 'STATION')
```

I then used *folium* to generate a basemap of New York City and added circle markers corresponding to the latitude and longitude coordinates of individual stations to it. I colored the markers as such:

* Red for the 5 busiest stations (based on total daily traffic)
* Orange for the second 5 busiest stations
* Yellow for the third 5 busiest stations
* Green for the fourth 5 busiest stations
* Blue for the fifth 5 busiest stations
* Black for all other stations

The code and map for this procedure are shown below:

```
import folium

m = folium.Map([40.75, -73.91], zoom_start=12)

for index, row in mta_join[mta_join.DATE=='2019-06-13'].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'black').add_to(m)
for index, row in mta_join[(mta_join.DATE=='2019-06-13') & 
                           (mta_join.STATION.isin(station_total.STATION[0:5]))].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'red').add_to(m)
for index, row in mta_join[(mta_join.DATE=='2019-06-13') & 
                           (mta_join.STATION.isin(station_total.STATION[5:10]))].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'orange').add_to(m)
for index, row in mta_join[(mta_join.DATE=='2019-06-13') & 
                           (mta_join.STATION.isin(station_total.STATION[10:15]))].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'yellow').add_to(m)
for index, row in mta_join[(mta_join.DATE=='2019-06-13') & 
                           (mta_join.STATION.isin(station_total.STATION[15:20]))].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'green').add_to(m)
for index, row in mta_join[(mta_join.DATE=='2019-06-13') & 
                           (mta_join.STATION.isin(station_total.STATION[25:30]))].iterrows():
    folium.CircleMarker([row['LAT'], row['LONG']], radius = 8, color = 'blue').add_to(m)

m.save('station_map.html')
```

![station_map](peterilhardt.github.io/images/station_map.png)

This confirmed that subway foot traffic is concentrated (almost exclusively) in downtown Manhattan, with some appearing in the Bay Ridge area. As such, it was safe to recommend Manhattan as the optimal target site for both gala recruitment and gala hosting. 

### Final Recommendations

Given all of the above information, I made the following recommendations to the client:

1. Target downtown Manhattan (and the busiest stations in particular) for visitor recruitment and gala placement. 
2. Because of the concentrated density of traffic at a select few stations, it would be better to place multiple street teams at the busiest stations (e.g. Grand Central, Penn Station, etc.) than spread the teams out to cover a larger area. 
3. Prioritize weekdays over weekends and holidays, especially the middle of the week.
4. Prioritize morning and midday over late afternoon/evening. This is because exiting pedestrians will be more likely to stop and talk to marketers than entering pedestrians, and exits appear to peak around midday. 
