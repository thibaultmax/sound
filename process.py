'''
	Copyright (C) 2019 Maxime Thibault

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
'''

import datetime
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# load the data files and concatenate them into a single pandas dataframe
datapath = os.path.join(os.getcwd(), 'data')
files_data = []
for file in os.listdir(datapath):
	file_df = pd.read_csv(os.path.join(datapath, file), sep='\t', index_col=0, names=['datetime', 'LEQ'], skiprows=3, usecols=[0,2], decimal=',')
	files_data.append(file_df)
all_data = pd.concat(files_data)

# process the concatenated dataframe to set the datetime as the index, get time descriptors as new columns (time only, date only, weekday, yearmonth, weekend),
# filter the data to keep only opening hours (+ 30 min before and after; based on 0800 to 2300 on weekdays and 0800 to 2100 on weekends), calculate overall L10 and L90, and make a second dataframe with L10, L50 and L90 for
# every day
all_data['dt_source'] = all_data.index.str.slice(start=0, stop=19)
all_data['dt_converted'] = pd.to_datetime(all_data['dt_source'], infer_datetime_format=True)
all_data.drop(columns=['dt_source'], inplace=True)
all_data.set_index('dt_converted', inplace=True)
all_data['time_only'] = all_data.index.time
all_data['dateonly'] = all_data.index.date
all_data['weekday'] = all_data.index.weekday
all_data['yearmonth'] = all_data.index.year.astype(str) + all_data.index.month.astype(str)
all_data['weekend'] = all_data['weekday'].apply(lambda x: 'week' if x<5 else 'weekend')
selected_data = all_data.loc[((all_data['weekday'] < 5 ) & (all_data['time_only'] >= datetime.time(7,30,0)) & (all_data['time_only'] <= datetime.time(23,59,0))) | ((all_data['weekday'] >= 5) & (all_data['time_only'] >= datetime.time(7,30,0)) & (all_data['time_only'] <= datetime.time(21,29,0)))].copy()
LEQ_L10_overall = selected_data['LEQ'].quantile(0.9)
LEQ_L50_overall = selected_data['LEQ'].quantile(0.5)
LEQ_L90_overall = selected_data['LEQ'].quantile(0.1)
selected_data['level_cat'] = selected_data['LEQ'].apply(lambda x: 'background' if x < LEQ_L90_overall else 'normal' if x < LEQ_L10_overall else 'annoying')
levels_per_day = selected_data.groupby(['dateonly'])['LEQ'].quantile([0.1,0.5,0.9]).reset_index()
levels_per_day['level_type'] = levels_per_day['level_1'].apply(lambda x: 'L90' if x == 0.1 else 'L50' if x == 0.5 else 'L10' if x == 0.9 else '')
levels_per_day['weekend'] = levels_per_day['dateonly'].apply(lambda x: 'week' if x.weekday() < 5 else 'weekend')
levels_per_day['yearmonth'] = levels_per_day['dateonly'].apply(lambda x: str(x.year) + str(x.month))

# make a graph of the frequency of levels above overall L10 at each hour during the day for every month
for name, group in selected_data.groupby(['yearmonth']):
	annoying_week = group.loc[(group['level_cat'] == 'annoying') & (group['weekend'] == 'week')].copy()
	annoying_weekend = group.loc[(group['level_cat'] == 'annoying') & (group['weekend'] == 'weekend')].copy()
	x_week = annoying_week.index.hour + (annoying_week.index.minute/60)
	x_weekend = annoying_weekend.index.hour + (annoying_weekend.index.minute/60)
	sns.set(style='darkgrid')
	sns.set_palette(sns.xkcd_palette(colors = ['windows blue', 'kelly green']))
	mal = sns.distplot(x_week, kde=True, rug=False, label='semaine')
	mal = sns.distplot(x_weekend, kde=True, rug=False, label='fin de semaine')
	mal.set(xlim=(7,25), xticks=[h for h in range(8,25,2)], xlabel='Heure', ylabel='Fréquence de bruit >L10')
	mal.legend()
	mal.figure.savefig(name[0] + 'annoyance_level.png')
	plt.gcf().clear()
	group_l10 = group['LEQ'].quantile([0.9])
	group_l50 = group['LEQ'].quantile([0.5])
	group_l90 = group['LEQ'].quantile([0.1])
	print('For month: {}  L10 was: {}   L50 was:  {}   L90 was: {}'.format(name[0], group_l10, group_l50, group_l90))

# make a graph plotting the l10, l50 and l90 levels of every month over time
for name, group in levels_per_day.groupby(['yearmonth']):
	sns.set(style='darkgrid')
	sns.set_palette(sns.xkcd_palette(colors = ['dark orange', 'windows blue', 'kelly green']))
	lpd = sns.relplot(x='dateonly', y='LEQ', kind='line', hue='level_type', data=group)
	lpd.set(xlabel='Date', ylabel='Niveau de son (dB-A)')
	lpd.fig.autofmt_xdate()
	lpd._legend.texts[0].set_text('Mesure')
	lpd.savefig(name[0] + 'levels_per_day.png')
	plt.gcf().clear()

	# make a graph plotting the distribution of l10, l50 and l90 of every day during the month
	sns.set(style='darkgrid')
	sns.set_palette(sns.xkcd_palette(colors = ['windows blue', 'kelly green', 'dark orange']))
	mal = sns.catplot(x='level_type', y='LEQ', data=group, kind='box')
	mal.set(ylabel='Niveau de son (db-A)', xlabel='Mesure')
	mal.savefig(name[0] + 'sound_level_dist.png')

# make a graph plotting the l10, l50 and l90 levels of every day over time
sns.set(style='darkgrid')
sns.set_palette(sns.xkcd_palette(colors = ['dark orange', 'windows blue', 'kelly green']))
lpd = sns.relplot(x='dateonly', y='LEQ', kind='line', hue='level_type', data=levels_per_day)
lpd.set(xlabel='Date', ylabel='Niveau de son (dB-A)')
lpd.fig.autofmt_xdate()
lpd._legend.texts[0].set_text('Mesure')
lpd.savefig('levels_per_day_all.png')
plt.gcf().clear()

# make two graphs comparing l10, l50 and l90 levels for one day compared to all other days
# and comparing the distribution of annoyance on this day vs all other days of the same type
limitedate = datetime.date(2019,7,31)
compdays = [limitedate - datetime.timedelta(days=x) for x in range(365)]
comptype = 'week'
comp_selected_data = selected_data.copy()
comp_levels_per_day = levels_per_day.copy()
comp_selected_data['group'] = comp_selected_data.apply(lambda x: 'before' if x.dateonly in compdays else 'after', axis=1)
comp_levels_per_day['group'] = comp_levels_per_day.apply(lambda x: 'before' if x.dateonly in compdays else 'after', axis=1)

sns.set(style='darkgrid')
sns.set_palette(sns.xkcd_palette(colors = ['dark orange', 'windows blue', 'kelly green']))
clpd = sns.catplot(x='group', y='LEQ', hue='level_type', data=comp_levels_per_day)
clpd.savefig('compared_levels_per_day.png')
plt.gcf().clear()

sns.set(style='darkgrid')
sns.set_palette(sns.xkcd_palette(colors = ['dark orange', 'windows blue', 'kelly green']))
cldpd = sns.catplot(x='group', y='LEQ', kind='box', data=comp_selected_data, showfliers=False)
cldpd.savefig('compared_LEQdist_per_day.png')
plt.gcf().clear()

annoying_before = comp_selected_data.loc[(comp_selected_data['level_cat'] == 'annoying') & (comp_selected_data['group'] == 'before')].copy()
annoying_after = comp_selected_data.loc[(comp_selected_data['level_cat'] == 'annoying') & (comp_selected_data['group'] == 'after')].copy()
x_compared = annoying_before.index.hour + (annoying_before.index.minute/60)
x_baseline = annoying_after.index.hour + (annoying_after.index.minute/60)
sns.set(style='darkgrid')
sns.set_palette(sns.xkcd_palette(colors = ['windows blue', 'kelly green']))
alc = sns.distplot(x_compared, kde=True, rug=False, label='before')
alc = sns.distplot(x_baseline, kde=True, rug=False, label='after')
alc.set(xlim=(7,25), xticks=[h for h in range(8,25,2)], xlabel='Heure', ylabel='Fréquence de bruit >L10')
alc.legend()
alc.figure.savefig('compared_annoyance_level.png')
plt.gcf().clear()


