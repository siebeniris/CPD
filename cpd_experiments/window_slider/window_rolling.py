import os
import json

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ruptures as rpt
from joblib import Parallel, delayed

from utils import *

matplotlib.use('Agg')


reno_keywords =['revamp', 'overhaul', 'remodel', 'update', 'redone', 'redo',
                'remodel', 'renovation', 
                'renovate', 'redecorate', 
                'refurbish', 'repair', 'refit', 'recondition',
                'renew', 'renewal', 'reform' ]

def preprocess_df(inputfile):
	df = pd.read_csv(inputfile)
	# change the date only to year-month-day
	df['date'] = df['date'].astype('datetime64[ns]', errors='ignore').dt.date
	# given the sentiment to output the polarity
	df.loc[df['sentiment']>=0.35, 'polarity']=1
	df.loc[df['sentiment']<= -0.35, 'polarity'] =-1
	df.loc[(df['sentiment']> -0.35) &(df['sentiment']<0.35), 'polarity'] =0

	lemmas = Parallel(n_jobs=-1)(delayed(lemmatize_sent)(sent) for sent in df.sentence.to_list())
	df['lemma'] = lemmas
	df['renovation'] = df.apply(lambda x :sum([ y in x.lemma for y in reno_keywords])>0, axis=1)

	return df
		
def get_cpd_df(df):
	# aggregate the dataframe by the date and its polarity scores for each date
	# across the uids.
	# eventually get the reviews by date.
	df_reno = df[df['renovation']==True]
	dates =[]
	scores =[]
	for date in sorted(list(set(df_reno.date.to_list()))):
		score =0
		
		rows = df_reno[df_reno['date']==date]
		for l in range(len(rows)):
			row = rows.iloc[l]
			score += row.polarity
		scores.append(score/len(rows))
		dates.append(date)
	
	# get the DataFrame by dates and scores
	df_cpd = pd.DataFrame(list(zip(dates, scores)), columns=['date', 'score'])
	return df_cpd

def rolling_window(df_cpd, outputpath, output_rpt, output_bkps):
	win1 = int(round(len(df_cpd)/15, 0))
	win2 = int(round(len(df_cpd)/5, 0))
	print(win1, win2)

	rolling_mean = df_cpd.score.rolling(window=win1).mean()
	rolling_mean2 = df_cpd.score.rolling(window=win2).mean()

	plt.figure(figsize=(40, 20))

	plt.plot(df_cpd.date, df_cpd.score, label='room')
	plt.plot(df_cpd.date, rolling_mean, label='window '+str(win1), color='orange')
	plt.plot(df_cpd.date, rolling_mean2, label='window '+str(win2), color ='magenta')

	plt.legend(loc='lower right')
	plt.savefig(outputpath)

	# detect the change point from rolling_mean2.
	y = np.array(rolling_mean2)
	x = np.array(df_cpd.date)[~np.isnan(y)].tolist()
	y = y[~np.isnan(y)]
	scores = y

	# the number of samples
	# rupture win model on the rolling mean 2.
	if win2>10 and isinstance(x, list) :
		plt.rcParams.update({'figure.max_open_warning': 0})

		model = "normal"
		algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(scores)
		my_bkps = algo.predict(n_bkps=3)
		rpt.show.display(scores, my_bkps, figsize=(10, 6))

		plt.title('Change Point Detection: Dynamic Programming Search Method')
		plt.savefig(output_rpt)
		plt.cla()
		plt.close()

		bkps = [int(x) for x in my_bkps]
		d = {
			'dates': x,
			'scores':scores.tolist(),
			'bkps': bkps
		}
		with open(output_bkps, 'w') as file:
			json.dump(d, file)
		
	


if __name__ == "__main__":
	data_path ='/home/yiyi/Documents/masterthesis/CPD/data'
	rolling_dir = os.path.join(data_path, 'rolling_window')
	rpt_dir = os.path.join(rolling_dir, 'ruptures')
	bkps_dir = os.path.join(rolling_dir, 'bkps')
	reno_dir = os.path.join(rolling_dir, 'renovation')
	plts = os.path.join(rolling_dir, 'rolling_window_plt')

	input_dir = os.path.join(data_path, 'sentiment_analysis', 'results')

	timer = Timer()

	for filename in os.listdir(input_dir):
		filepath = os.path.join(input_dir, filename)
		if os.path.isfile(filepath):
			idx = int(filename.split('#')[0])
			timer.start()

			print('processing ', filename,' ...')
			try:
			
				reno_file = os.path.join(reno_dir, filename)
				if not os.path.exists(reno_file):
					df = preprocess_df(filepath)
					df.to_csv(reno_file)
				else:
					df = pd.read_csv(reno_file)

				bkps_filepath = os.path.join(bkps_dir, filename+'.json')
				try:
					if os.path.exists(bkps_filepath) and os.path.isfile(bkps_filepath):
						with open(bkps_filepath) as file:
							bkps_dict = json.load(file)
						bkps = bkps_dict['bkps']
						print(bkps)
						assert len(bkps)==4
				except Exception:
					df_cpd = get_cpd_df(df)
					rolling_window(df_cpd, os.path.join(plts, filename+'.png'), 
							os.path.join(rpt_dir, filename+'.png'),
							bkps_filepath)
			except Exception:
				print(filename+' problem...')
			
			timer.stop()


