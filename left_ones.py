import random,pickle
import numpy as np
import matplotlib.pylab as plt
from rbm import *
from denoising_autoencoder import dA
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
from numpy import array as ar
import pdb
import distance
import os
from ga import KnapsackData
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class LeftOnes(object):
    """docstring for LeftOnes"""

    def __init__(self,corruption_level=0.2):
        super(LeftOnes, self).__init__()
        print self
        self.test = 5
        self.dA = dA(n_visible=20,n_hidden=50)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        
    def fitness(self,string):
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        return fitness

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def generate_random_string(self,l=20):
        [random.choice([0,1]) for i in range(l)]

    def get_good_strings(self,strings,lim=20):
        print strings[0:5]
        fitnesses =  [self.fitness(s) for s in strings]
        # print fitnesses[0:10]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        return [strings[i] for i in sorted_fitnesses[0:lim]],sorted_fitnesses[0:lim]

    def generate_good_strings(self,x=1000,l=20,lim=20):
        strings = [[random.choice([0,1]) for i in range(l)] for _ in range(x)]
        fitnesses =  [self.fitness(s) for s in strings]
        sorted_fitnesses = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
        sorted_fitnesses.reverse()
        return strings,[strings[i] for i in sorted_fitnesses[0:lim]]

    def train_dA(self,data,corruption_level=0.2,num_epochs=200):
        train_data = data
        # pdb.set_trace()
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=0.1,num_epochs=num_epochs)

    def build_sample_dA(self):
        self.sample_dA = theano.function([self.dA.input],self.dA.sample)

    def calculate_changes_in_fitness(self,population,number_of_trials):
        original_fitnesses = ar(lo.fitness_many(population))
        print original_fitnesses.shape
        sample = [lo.sample_dA([i]) for i in population]
        # print sample.shape
        sample_fitnesses = ar(lo.fitness_many([j[0] for j in sample]))
        # return original_fitnesses,sample,sample_fitnesses
        print sample_fitnesses.shape
        print sample_fitnesses[0:10]
        differences = sample_fitnesses - original_fitnesses
        distances = [[distance.hamming(population[k],sample[k][0]) for k in range(len(sample))]]
        # pdb.set_trace()
        for i in range(number_of_trials):
            print "trial:",i
            new_sample = [lo.sample_dA([j]) for j in population]
            new_sample_fitnesses = ar(lo.fitness_many([j[0] for j in new_sample]))
            new_difference = new_sample_fitnesses - original_fitnesses
            sample_fitnesses = np.vstack((sample_fitnesses,new_sample_fitnesses))
            differences = np.vstack((differences,new_difference))
            distances.append([distance.hamming(population[k],sample[k][0]) for k in range(len(sample))])
        return sample_fitnesses,differences,distances

    def get_statistics(self,population,sample):
        original_fitnesses = ar(lo.fitness_many(population))
        sample_fitnesses = ar(lo.fitness_many([j[0] for j in sample]))
        differences = sample_fitnesses - original_fitnesses
        distances = [distance.hamming(population[k],sample[k][0]) for k in range(len(sample))]
        return original_fitnesses,sample_fitnesses,differences,distances

    def experiment(self,name,no_trials=10,corruption_level=0.2):
        ensure_dir("results/autoencoder/".format(name))
        all_strings,good_strings=self.generate_good_strings(10000)
        self.train_dA(ar(good_strings),corruption_level=corruption_level)
        original_fitnesses = self.fitness_many(all_strings)
        f,d,dist = lo.calculate_changes_in_fitness(all_strings,no_trials)
        data = {
        "original":original_fitnesses,
        "fitnesses_sampled":f,
        "differences_in_fitness":d,
        "distances":dist,
        "no_trials":no_trials,
        "corruption_level":corruption_level,
        "all_strings":all_strings,
        "good_strings":good_strings
        }
        pickle.dump(data,open("results/autoencoder/{0}.pkl".format(name),"wb"))
        return data

    def iterative_algorithm(self,name,pop_size=100,genome_length=20,lim=20,trials=10,corruption_level=0.2,num_epochs=50):
        fitfile = open("{0}_fitnesses.dat".format(name),"w")
        self.dA = dA(n_visible=genome_length,n_hidden=50)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,lim)
        self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=50)
        sampled_population = [lo.sample_dA([i]) for i in all_strings]
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,sampled_population)
        data = {
        "original":original_fitnesses,
        "fitnesses_sampled":sample_fitnesses,
        "differences_in_fitness":differences,
        "distances":distances
        }
        pickle.dump(data,open("results/autoencoder/{0}_0.pkl".format(name),"wb"))
        fitfile.write("{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses)))
        fitfile.write("{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        for iteration in range(0,trials):
            population = [z[0] for z in sampled_population]
            self.dA = dA(n_visible=genome_length,n_hidden=50)
            self.dA.build_dA(corruption_level)
            self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,lim)
            self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs)
            sampled_population = [lo.sample_dA([i]) for i in population]
            original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,sampled_population)
            data = {
            "original":original_fitnesses,
            "fitnesses_sampled":sample_fitnesses,
            "differences_in_fitness":differences,
            "distances":distances
            }
            pickle.dump(data,open("results/autoencoder/{0}_{1}.pkl".format(name,iteration+1),"wb"))
            fitfile.write("{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses)))
            fitfile.flush()
            print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        fitfile.close()

class ACSKnapsack(LeftOnes):
    """docstring for ACSKnapsack"""
    def __init__(self, corruption_level=0.2,knapsack_file="weing1.pkl"):
        super(ACSKnapsack, self).__init__(corruption_level=corruption_level)
        self.knapsack = pickle.load(open(knapsack_file))

    def fitness(self,string):
        knapsack = self.knapsack
        weights = []
        for i,c in enumerate(knapsack.capacities):
            weights.append(np.sum(np.array(knapsack.constraints[i])*string))
        over = 0
        for i,w in enumerate(weights):
            if w > knapsack.capacities[i]:
                over += (w - knapsack.capacities[i])
        if over > 0:
            return -over
        else:
            return np.sum(np.array(knapsack.values)*string)
        


if __name__ == '__main__':
    # all_strings,good_strings=lo.generate_good_strings(10000)
    # lo.fitness_many(good_strings)
    # lo.train_dA(ar(good_strings))
    # z = [lo.sample_dA([i]) for i in all_strings]
    # plt.subplot(1,2,2)
    # plt.title("new")
    # plt.hist(lo.fitness_many([i[0] for i in z]))
    # plt.subplot(1,2,1)
    # plt.title("orginal")
    # plt.hist(lo.fitness_many(all_strings))
    # plt.show()
    # f,d,dist = lo.calculate_changes_in_fitness(all_strings,10)
    c_level = 0.1
    # lo = LeftOnes(corruption_level=c_level)
    # data =lo.experiment("c-{0}".format(c_level),no_trials=100,corruption_level=c_level)
    # t= Test()
    # lo.iterative_algorithm("test_p_100_0.2",trials=20,corruption_level=0.2)
    # data=lo.fitness_many(all_strings)
    # y,binEdges=np.histogram(data,bins=100)
    # bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    # p.plot(bincenters,y,'-')
    # p.show()
    for i in range(0,10):
        lo = ACSKnapsack(corruption_level=c_level,knapsack_file="weing8.pkl")
        # data =lo.experiment("c-{0}".format(c_level),no_trials=100,corruption_level=c_level)
        # t= Test()
        lo.iterative_algorithm("top_20_epochs_50_knapsack_{0}".format(i),pop_size=10000,genome_length=105,lim=20,trials=50,corruption_level=0.2,num_epochs=50)
