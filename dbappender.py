import os
import sys
import MySQLdb
import librosa
import redis
import signal
import time
import numpy as np
import os
r_server=redis.Redis('localhost')

class timeout(Exception):
	pass

def handler(signum,frame):
	print('Time limit exceeded')
	raise timeout

def mapper(mfcc):
	return mfcc[-1292:]

def _already_computed(emotion,computed_list,error):
	for foo in r_server.lrange(emotion,0,-1):
		computed_list.append(foo[len(emotion)+1:])
	for foo in r_server.lrange("problem_"+emotion,0,-1):
		error.append(foo)




signal.signal(signal.SIGALRM,handler)

def main():
	add_emotion=[]
	add_path=[]
	training_dataset=os.getcwd()+'/training_dataset'
	filename=os.listdir(training_dataset)
	for foo in filename:
		add_emotion.append(foo)
		add_path.append(training_dataset+'/'+foo)
	for foo in range(0,len(add_path)):
		computed_list=[]
		error=[]
		_already_computed(add_emotion[foo],computed_list,error)
		file_names=os.listdir(add_path[foo])
		for file_name in file_names:
			if file_name in computed_list:
				continue
			if file_name in error:
				continue
			signal.alarm(60*2)
			try:
				print(file_name+" computation begins")
				y,sr= librosa.load(add_path[foo]+'/'+file_name)
				print("y sr computed")
				mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
				mfcc=np.array(map(mapper,mfcc))
				mfcc=mfcc.max(axis=0)
				print("mfcc computed")
				r_server.rpush(add_emotion[foo],add_emotion[foo]+'_'+file_name)
				for foo1 in mfcc:
					r_server.rpush(add_emotion[foo]+'_'+file_name,foo1)
			except timeout:
				print("error")
				r_server.lpush("problem_"+add_emotion[foo],file_name)
			else:
				signal.alarm(0)
				print("added")



main()

