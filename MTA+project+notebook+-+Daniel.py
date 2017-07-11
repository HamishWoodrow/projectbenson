
# coding: utf-8

# In[233]:

# imports a library 'pandas', names it as 'pd'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from IPython.display import Image

# enables inline plots, without it plots don't show up in the notebook
#get_ipython().magic('matplotlib inline')


# ### load Jin's pre-cleaned data

# In[234]:


#df = pd.read_csv("/Users/dlicht/ds/metis/metisgh/DataSets/MTA/month_data_updated.csv")
df = pd.read_csv("/Users/dlicht/ds/metis/metisgh/DataSets/MTA/clean_data.csv")



# In[235]:

df.columns


# In[4]:

df.head()


# In[236]:

df[['entries','exits']].describe()


# In[278]:

df.dtypes


# In[289]:

#convert datetime column to datetime object
#from dateutil.parser import *
from datetime import datetime

#df['datetime_ob'] = df["parsed_datetime"].map(lambda t: parse(t))
df['datetime_ob'] = df["parsed_datetime"].map(lambda t: datetime.strptime(t,'%Y-%m-%d %X'))


# In[319]:

df.head(400)


# In[291]:

# confirm negative/zero values removed (this is Jin's code)
print('percentage of negative entries', "{0:.3f}%".format(df[df.entries < 0].shape[0]/df.shape[0]))
print('percentage of zero entries', "{0:.3f}%".format(df[df.entries == 0].shape[0]/df.shape[0]))
print('percentage of negative exits', "{0:.3f}%".format(df[df.exits < 0].shape[0]/df.shape[0]))
print('percentage of zero entries', "{0:.3f}%".format(df[df.exits == 0].shape[0]/df.shape[0]))

# - since negative percentage is small, we can take a look at the negative numbers to tell how the numbers are wrong
# - the zero recordings are likely to be from the station closing
# - we can manually get a sense the outlieres of positive numbers by eyeballing the numbers


# In[293]:

plt.hist(df.entries,bins = 100, range=(0,20000),log=True);


# ## reject outliers greater than 15000 (if any left)

# In[294]:

df_15K_reject = df[df.entries<15000]
df_15K_reject = df_15K_reject[df_15K_reject.exits<15000]


# In[295]:

plt.figure(figsize=(10, 8))
plt.hist(df_15K_reject.exits,bins = 300, range=(0,15000),log=True);


# In[296]:

#total up per station
sdf = df_15K_reject.groupby('station').sum()
sdf.head()


# In[297]:

#total station activity
sEE = sdf[['entries','exits']]
sEE


# In[298]:

plt.hist(sEE.entries)


# In[299]:


fig, ax = plt.subplots(figsize=(10, 60))
y_pos = np.arange(len(sEE.index))
ax.set_yticks(y_pos)
ax.set_yticklabels(sEE.index,fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('total ridership')
ax.set_title('Station Utilization')
plt.barh(y_pos,(sEE.entries+sEE.exits)/2,log=False);
#plt.barh(sEE.index,sEE.entries,log=False)


# In[301]:

sEE.index


# ## find the 20 busiest stations

# In[302]:

sEE['activity'] = sEE.apply(lambda x: (x['entries']+x['exits']),axis=1)

#weather_temps.apply(lambda x: x['Temp (C)'] - 2*x['Temp (F)'],axis=1)
sEE


# In[304]:

sorted_sEE = sEE.sort_values('activity',ascending=False)
sorted_sEE.head()


# In[305]:

fig, ax = plt.subplots(figsize=(10, 60))
y_pos = np.arange(len(sorted_sEE.index))
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_sEE.index,fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('total ridership')
ax.set_title('Station Utilization')
plt.barh(y_pos,(sorted_sEE.entries+sorted_sEE.exits)/2,log=False);


# In[306]:

sorted_sEE.iloc[0:20]


# In[307]:

TopStations=sorted_sEE.iloc[0:20]
TopStations.index


# In[308]:

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(TopStations.index))
ax.set_yticks(y_pos)
ax.set_yticklabels(TopStations.index)
ax.invert_yaxis()
ax.set_xlabel('total ridership')
ax.set_title('Station Utilization')
plt.barh(y_pos,(TopStations.entries+TopStations.exits)/2,log=False);



# In[309]:


fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(TopStations.index))
ax.set_yticks(y_pos)
ax.set_yticklabels(TopStations.index)
ax.invert_yaxis()
ax.set_xlabel('total ridership')
ax.set_title('Entrance-Exit difference for top 20 busiest stations')
plt.barh(y_pos,(TopStations.entries-TopStations.exits),log=False);



# In[ ]:




# In[310]:

df_15K_reject.columns


# In[322]:

df_station_hourly = df_15K_reject.groupby(['station','datetime_ob']).sum()

df_station_hourly.head()


# In[324]:

df_station_hourly.sample(20)


# In[325]:

df_station_hourly


# In[327]:

#looking at the entry data for one station only
st23 = df_station_hourly.loc['23 ST'].entries
st23.head(10)


# In[328]:

st23.index


# In[330]:

time = st23.index


#from dateutil.parser import *
#time = [parse(t) for t in firstAV.index]
plt.figure(figsize=(20, 4))
plt.plot(time,st23);

#NOTE THE MISSING PEAK ON MAY 29TH, THAT WAS MEMORIAL DAY


# In[331]:

time


# In[332]:

import datetime
st23_hr = st23.index.map(lambda t: t.hour)
st23_hr


# In[333]:

plt.figure(figsize=(10, 4))
plt.plot(st23_hr,st23);


# In[334]:

#looking at the entry data for one station only
st23_cum_enter = df_station_hourly.loc['23 ST'].cum_entries
st23_cum_enter.head(10)


# In[335]:

plt.figure(figsize=(10, 4))
plt.plot(time,st23_cum_enter);


# In[ ]:




# In[ ]:




# In[ ]:




# In[336]:

df_station_time = df_15K_reject.groupby(['station','time']).sum()

df_station_time.head(10)


# In[340]:

#what % of ridership happened during odd time intervals?
#(not multiples of 4hrs)


# In[341]:

df = df_15K_reject
df_station_ODDtime = df[(df.time!='00:00:00')&(df.time!='04:00:00')                       &(df.time!='08:00:00')&(df.time!='12:00:00')                       &(df.time!='16:00:00')&(df.time!='20:00:00')]                        .groupby(['station','time']).sum()
df_station_ODDtime.head(100)


# In[342]:

df_station_ODDtime.sum()


# In[343]:

df_station_time.sum()


# In[344]:

PercentIrregular =  (6.227399e+07+4.850229e+07)/(1.396210e+08+1.071474e+08)
PercentIrregular


# If 45% of the data is found in irregular time entries.  Some type of allowance for this will need to be made...

# In[57]:

## needs a riders thus far that day column.


# In[375]:

df_hourly = df_15K_reject.groupby(['station','time']).sum()
df_hourly.head()


# In[392]:

df_hourly = df_hourly.reset_index(level=1)
df_hourly.head()


# In[404]:

from datetime import datetime
df_hourly['time_ob'] = df_hourly["time"].map(lambda t: datetime.strptime(t,'%X'))
df_hourly.dtypes


# In[405]:

df_hourly.head()


# In[471]:

#summing up all data that starts between 2am and 6am to make a 4am time window value

start_window = datetime.strptime("02:00:00",'%X')
print(start_window)
end_window   = datetime.strptime("06:00:00",'%X')
print(end_window)

#df_15K_reject = df[df.entries<15000]
temp = df_hourly[df_hourly.time_ob>=start_window]
df04 = temp[temp.time_ob<end_window]
df04.head()


# In[472]:

df04sum = df04.sum()
df04sum
df04sum_val = (df04sum.entries + df04sum.exits)/2
df04sum_val


# In[473]:

hourly_sum_list = []
hourly_sum_list += [df04sum_val]
hourly_sum_list


# In[474]:

#summing up all data that starts between 6am and 10am to make a 8am time window value

start_window = datetime.strptime("06:00:00",'%X')
end_window   = datetime.strptime("10:00:00",'%X')

temp = df_hourly[df_hourly.time_ob>=start_window]
df08 = temp[temp.time_ob<end_window]
df08.head()
df08sum = df08.sum()
df08sum
df08sum_val = (df08sum.entries + df08sum.exits)/2
df08sum_val
hourly_sum_list += [df08sum_val]
hourly_sum_list


# In[475]:

#summing up all data that starts between 10:00 and 14:00 to make a 12:00 time window value

start_window = datetime.strptime("10:00:00",'%X')
end_window   = datetime.strptime("14:00:00",'%X')

temp = df_hourly[df_hourly.time_ob>=start_window]
df12 = temp[temp.time_ob<end_window]
df12.head()
df12sum = df12.sum()
df12sum
df12sum_val = (df12sum.entries + df12sum.exits)/2
df12sum_val
hourly_sum_list += [df12sum_val]
hourly_sum_list


# In[476]:

#summing up all data that starts between 14:00 and 18:00 to make a 16:00 time window value

start_window = datetime.strptime("14:00:00",'%X')
end_window   = datetime.strptime("18:00:00",'%X')

temp = df_hourly[df_hourly.time_ob>=start_window]
df16 = temp[temp.time_ob<end_window]
df16.head()
df16sum = df16.sum()
df16sum
df16sum_val = (df16sum.entries + df16sum.exits)/2
df16sum_val
hourly_sum_list += [df16sum_val]
hourly_sum_list


# In[477]:

#summing up all data that starts between 18:00 and 22:00 to make a 20:00 time window value

start_window = datetime.strptime("18:00:00",'%X')
end_window   = datetime.strptime("22:00:00",'%X')

temp = df_hourly[df_hourly.time_ob>=start_window]
df20 = temp[temp.time_ob<end_window]
df20.head()
df20sum = df20.sum()
df20sum
df20sum_val = (df20sum.entries + df20sum.exits)/2
df20sum_val
hourly_sum_list += [df20sum_val]
hourly_sum_list


# In[478]:

#summing up all data that starts between 22:00 and 02:00 to make a 00:00 time window value


#note this one is two separated ranges

start_window = datetime.strptime("22:00:00",'%X')
end_window   = datetime.strptime("02:00:00",'%X')

temp1 = df_hourly[df_hourly.time_ob>=start_window]
temp2 = df_hourly[df_hourly.time_ob<end_window]

df00sum_late = temp1.sum()
df00sum_early = temp1.sum()

df00sum_val = (df00sum_late.entries + df00sum_late.exits)/2 + (df00sum_early.entries + df00sum_early.exits)/2
df00sum_val
hourly_sum_list += [df00sum_val]
hourly_sum_list


# In[479]:

hourly_sum_list


# In[480]:




# In[ ]:




# In[486]:

timelist = ['4:00','8:00','12:00','16:00','20:00','00:00']
timelist
fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(len(timelist))

ax.set_xticks(x_pos)
#ax.set_xticklabels(sdf_weekday.index)
ax.set_xticklabels(timelist)
ax.set_xlabel('Time of Day')
ax.set_ylabel('Total Ridership')
ax.set_title('Total Ridership by Time of Day')
ax.set_autoscaley_on(False)
ax.set_ybound(lower=0,upper=5e8)

plt.scatter(x_pos,hourly_sum_list)
plt.plot(x_pos,hourly_sum_list);


# In[ ]:




# In[ ]:




# In[469]:

df20


# In[467]:

df08sum


# In[470]:

df20sum


# In[ ]:




# In[ ]:




# # day of the week analysis

# In[348]:

df_15K_reject["weekday"] = df_15K_reject['datetime_ob'].map(lambda t: t.weekday())
#weather['Temp (F)'] = weather['Temp (C)'].map(lambda x: 9.0*x/5.0 + 32)
df_15K_reject.head()


# In[349]:

#total up entire system per day of the week
sdf_weekday = df_15K_reject.groupby(['weekday']).sum()
sdf_weekday.head(9)


# In[85]:

fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(len(sdf_weekday.index))
ax.set_xticks(x_pos)
#ax.set_xticklabels(sdf_weekday.index)
ax.set_xticklabels(['M','T','W','Th','F','Sa','Su'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('total ridership')
ax.set_title('Total Ridership by Day of the Week')

plt.bar(sdf_weekday.index, (sdf_weekday.entries+sdf_weekday.exits)/2);


# In[350]:

fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(len(sdf_weekday.index))
ax.set_xticks(x_pos)
#ax.set_xticklabels(sdf_weekday.index)
ax.set_xticklabels(['M','T','W','Th','F','Sa','Su'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('total ridership')
ax.set_title('Total Ridership by Day of the Week')

plt.plot(sdf_weekday.index, (sdf_weekday.entries+sdf_weekday.exits)/2);


# In[351]:

#total up per station, per day of the week
sdf_weekday = df_15K_reject.groupby(['station','weekday']).sum()
sdf_weekday.head(9)


# In[352]:

#sdf_weekday.loc[["86 ST","23 ST"]]
top20_weekday = sdf_weekday.loc[list(TopStations.index)]
top20_weekday.loc['14 ST']


# In[360]:

fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(7)
ax.set_xticks(x_pos)
#ax.set_xticklabels(sdf_weekday.index)
ax.set_xticklabels(['M','T','W','Th','F','Sa','Su'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Avg Daily Ridership')
ax.set_title('Total Ridership by Day of the Week')

#leg = plt.figlegend(top20_weekday.loc[station].index, TopStations.index, loc=(0.85, 0.65))
#ax.legend(top20_weekday.loc[station].index)

nweeks = 52
for station in TopStations.index:
    plt.plot(top20_weekday.loc[station].index,              (top20_weekday.loc[station].entries+top20_weekday.loc[station].exits)/(2*nweeks),            label = station)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
("")


# In[361]:

Avg_daily_station_usage = (TopStations.entries+TopStations.exits)/(2*nweeks*7)


fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(7)
ax.set_xticks(x_pos)
#ax.set_xticklabels(sdf_weekday.index)
ax.set_xticklabels(['M','T','W','Th','F','Sa','Su'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Normalized ridership')
ax.set_title('NORMALIZED Ridership by Day of the Week \n Top 20 Stations')

#leg = plt.figlegend(top20_weekday.loc[station].index, TopStations.index, loc=(0.85, 0.65))
#ax.legend(top20_weekday.loc[station].index)

for station in TopStations.index:
    plt.plot(top20_weekday.loc[station].index,              (top20_weekday.loc[station].entries+top20_weekday.loc[station].exits)/(2*nweeks*Avg_daily_station_usage[station]),            label = station)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
("")


# ### weekdays vs. weekends

# In[362]:

df_15K_reject["weekend"] = df_15K_reject['weekday'].map(lambda t: t>=5)
#weather['Temp (F)'] = weather['Temp (C)'].map(lambda x: 9.0*x/5.0 + 32)
df_15K_reject.head()


# In[363]:

sdf_weekend = df_15K_reject.groupby(['station','weekday','weekend']).sum()
sdf_weekend = sdf_weekend.groupby(['station','weekend']).mean()  #note mean() this time instead of sum()
sdf_weekend.head(9)


# In[364]:

top20_weekend = sdf_weekend.loc[list(TopStations.index)]
top20_weekend.loc['14 ST']


# In[365]:

fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(7)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Weekday','Weekend'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Avg Daily Ridership')
ax.set_title('Total Ridership by Day of the Week')

for station in TopStations.index:
    plt.scatter(top20_weekend.loc[station].index+0.2*np.random.random()-0.1,              (top20_weekend.loc[station].entries+top20_weekend.loc[station].exits)/(2),            label = station)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));


# In[366]:

fig, ax = plt.subplots(figsize=(10, 7))
x_pos = np.arange(7)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Weekday','Weekend'])
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Avg Daily Ridership')
ax.set_title('Total Ridership by Day of the Week')

for station in TopStations.index:
    plt.scatter(top20_weekend.loc[station].index,              (top20_weekend.loc[station].entries+top20_weekend.loc[station].exits)/(2),            label = station)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
for station in TopStations.index:
    plt.plot(top20_weekend.loc[station].index,              (top20_weekend.loc[station].entries+top20_weekend.loc[station].exits)/(2),            label = station)
("")


# In[ ]:




# In[367]:

top20_weekend.loc['23 ST'].index


# In[368]:

top20_weekend.loc[station].index


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:
