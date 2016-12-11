f1 = open('../features/usertopics.txt', 'w')

with open('../train_data/user_info.txt', 'r') as f2:
	for line in f2:
		topics = []
		try:
			topics = line.split()[1].split('/')
			uid = line.split()[0]
		except:
			pass
		for topic in topics:
			f1.write(uid+','+topic+'\n')
			