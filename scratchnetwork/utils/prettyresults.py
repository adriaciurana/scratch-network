import math, os
class PrettyResults(object):
	def __init__(self, fullscreen, margin=1):
		self.fullscreen = fullscreen
		self.rows = []
		self.margin = margin

	def reset(self):
		self.rows = []

	def add_row(self, row='', spacer=' ', is_dict=False):
		self.rows.append((str(row), spacer, is_dict))

	def add_dictionary(self, dictionarys, spacer_keys=' ', spacer_values=' '):
		def __add_dict(dictionary):
			keys = map(lambda x: str(x), dictionary.keys())
			values = map(lambda x: str(x), dictionary.values())
			str_keys = ''
			str_values = ''
			for k, v in zip(keys, values):
				len_k = len(k)
				len_v = len(v)
				if len_k > len_v:
					margin = (len_k - len_v) / 2
					margin_1 = margin - 1
					margin_1a = math.ceil(margin_1)
					margin_1b = math.floor(margin_1)
					v = margin_1a*spacer_values + (margin_1 >= 0)*' ' + v + (margin_1 >= 0)*' ' + margin_1b*spacer_values
				else:
					margin = (len_v - len_k) / 2
					margin_1 = margin - 1
					margin_1a = math.ceil(margin_1)
					margin_1b = math.floor(margin_1)
					k = margin_1a*spacer_keys + (margin_1 >= 0)*' ' + k + (margin_1 >= 0)*' ' + margin_1b*spacer_keys
				str_keys += k + ' | '
				str_values += v + ' | '

			self.add_row(str_keys[:-3], spacer=spacer_keys, is_dict=True)
			self.add_row(str_values[:-3], spacer=spacer_values, is_dict=True)

		if isinstance(dictionarys, dict):
			__add_dict(dictionarys)
		else:
			for dictionary in dictionarys:
				__add_dict(dictionary)



	def print(self):
		if self.fullscreen:
			max_len = int(os.popen('stty size', 'r').read().split()[1]) - 2
		else:
			max_len = max(map(lambda row: len(row[0]), self.rows)) + 2*self.margin
		print((max_len + 2)*'=')
		for row, spacer, is_dict in self.rows:
			margin = (max_len - len(row)) / 2
			if is_dict:
				margin_a = math.ceil(margin)
				margin_b = math.floor(margin)
				str_r = '|' + margin_a*spacer + row + margin_b*spacer + '|'
			else:
				margin_1 = margin - 1
				margin_1a = math.ceil(margin_1)
				margin_1b = math.floor(margin_1)
				str_r = '|' + margin_1a*spacer + (margin_1 >= 0)*' ' + row + (margin_1 >= 0)*' ' + margin_1b*spacer + '|'
			print(str_r)
		print((max_len + 2)*'=')
