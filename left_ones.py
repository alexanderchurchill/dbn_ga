import random
import numpy as np
import matplotlib.pylab as plt

def fitness(string):
	fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
	return fitness

def fitness_many(strings):
	return [fitness(s) for s in strings]

def generate_random_string(l=20):
	[random.choice([0,1]) for i in range(l)]

def generate_good_strings(x=1000,l=20,lim=20):
	strings = [[random.choice([0,1]) for i in range(l)] for _ in range(x)]
	fitnesses =  [fitness(s) for s in strings]
	sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
	sorted_fitnesses.reverse()
	return strings,[strings[i] for i in sorted_fitnesses[0:lim]]

all_strings,good_strings=generate_good_strings(1000)
fitness_many(good_strings)