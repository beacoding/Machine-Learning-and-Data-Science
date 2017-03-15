# {'predict': <function predict at 0x7fdbcd8e3b90>, 

# 'splitModel': {'predict': <function predict at 0x7fdbcd8e39b0>, 'splitValue': -122.0, 'splitNot': 2, 'splitSat': 1, 'splitVariable': 0}, 
# 	'subModel0': {'splitNot': 1, 'splitSat': 2, 'splitModel': {...}, 'predict': <function predict at 0x7fdbcd8e39b0>, 'splitVariable': 0, 'splitValue': -167.0}, 
# 	'subModel1': {'splitNot': 1, 'splitSat': 2, 'splitModel': {...}, 'predict': <function predict at 0x7fdbcd8e39b0>, 'splitVariable': 1, 'splitValue': 45.0}}

if longitude > -122.0:
	if latitude > 45.0:
		return 2
	else:
		return 1
else:
	if longitude > -167.0:
		return 2
	else:
		return 1


