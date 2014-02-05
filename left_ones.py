import random
import numpy as np
import matplotlib.pylab as plt
from rbm import *
from denoising_autoencoder import dA
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
from numpy import array as ar
import pdb
class LeftOnes(object):
    """docstring for LeftOnes"""

    def __init__(self):
        super(LeftOnes, self).__init__()
        print self
        self.test = 5
        self.dA = dA(n_visible=20,n_hidden=50)
        self.dA.build_dA(0.2)
        self.build_sample_dA()
        
    def fitness(self,string):
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        return fitness

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def generate_random_string(self,l=20):
        [random.choice([0,1]) for i in range(l)]

    def generate_good_strings(self,x=1000,l=20,lim=20):
        strings = [[random.choice([0,1]) for i in range(l)] for _ in range(x)]
        fitnesses =  [self.fitness(s) for s in strings]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        return strings,[strings[i] for i in sorted_fitnesses[0:lim]]

    def train_dA(self,data,corruption_level=0.2):
        train_data = data
        # pdb.set_trace()
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=0.1,num_epochs=200)

    def build_sample_dA(self):
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

if __name__ == '__main__':
    lo = LeftOnes()
    all_strings,good_strings=lo.generate_good_strings(10000)
    lo.fitness_many(good_strings)
    lo.train_dA(ar(good_strings))
    # t= Test()


