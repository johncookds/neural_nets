import numpy as np
from collections import defaultdict
from copy import deepcopy

class mutate_top(object):
	def __init__(self, nn_class,attributes=[]):
		self.nn_class=nn_class
		#attributes to perturb
		self.attributes=attributes

	def pass_params(self,**kwargs):
		self.nn_class.set_params(kwargs=kwargs['kwargs'])

	def random_population(self, shape):
		return np.array([np.array(shape),1])

	def _exrink(self, listt,length): #expand or shrink
		if len(listt)>=length:
			return listt[:length]
		else:
			return listt + [listt[-1]]*(length-len(listt))

	def perturb(self,organism,scale=False,noise=3.0):
		for attr in self.attributes:
			setting=organism['attributes'][attr]
			if type(setting['value'])==list:
				setting['value']=self._exrink(setting['value'],len(setting['value'])+int(np.random.normal(scale=setting['scale'])))
				for lvl in setting['value']:
					sc=np.random.normal(scale=noise)
					lvl['value'] = max(32,int(lvl['value']+lvl['scale']*int(sc)))
			else:
				setting['value'] = int(setting['value']+setting['scale']*int(np.random.normal(scale=noise)))
		return organism

	def _dict_check(self, child, children):
		for ch in children:
			same=True
			for attr in self.attributes:
				if ch['attributes'][attr]['value']!=child['attributes'][attr]['value']:
					same=False
			if same==True:
				return False
		return False

	def top_survives(self,fit_sorted,num_children=5):
		winner=fit_sorted[0]
		children=[]
		while len(children) <num_children:
			new_child=self.perturb(winner)
			children.append(deepcopy(new_child))
		return children

	def _reformat(self,d):
		for key in d.keys():
			if type(d[key])==dict:
				d[key]=[v['value'] for v in d[key]['value']]
		return d

	def test_individuals(self,population):
		fit=[]
		for org in population:
			attr=deepcopy(org['attributes'])
			self.nn_class.set_params(kwargs=self._reformat(attr))
			fitness=self.nn_class.run()
			fit.append([org,fitness])
		fit_sorted=sorted(fit,key=lambda x: x[1], reverse=True)
		return fit_sorted
#	Original = {'attributes': {'n_inputs': 784, 
#	'hidden_lvls': {'value': [{'value': 284, 'scale': 8.0},{'value':284,'scale':8.0}],'scale':0} #setting scale=0 keeps same number of hidden levels
#	'n_classes':10 }}

	def evolve(self, original, generations=10):
		pop_fitness=[[original,0.0]]
		gen=0
		fit_levels=defaultdict(list)
		while gen<generations:
			pop=self.top_survives(pop_fitness[0])
			pop_fitness=self.test_individuals(pop)
			fit_levels[gen].append(pop_fitness)
			gen+=1
		return fit_levels