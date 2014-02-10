import random,pickle,copy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from rbm import *
from denoising_autoencoder import dA
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
from numpy import array as ar
import pdb
import distance
import os,sys
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

    def weighted_random_choice(self,population,fitness_sum):
        pick = random.uniform(0, fitness_sum)
        current = 0
        for chromosome in population:
            current += self.fitness(chromosome)
            if current > pick:
                return copy.deepcopy(chromosome)

    def get_new_population_rw(self,population, pop_size=None):
        if pop_size == None:
            pop_size = len(population)
        sum_fitnesses = sum(self.fitness_many(population))
        new_population = [self.weighted_random_choice(population,sum_fitnesses)
                            for p in range(pop_size)]
        return population

    def fitness(self,string):
        fitness = sum(string[0:len(string)/2]) - sum(string[len(string)/2:])
        return fitness

    def fitness_many(self,strings):
        return [self.fitness(s) for s in strings]

    def generate_random_string(self,l=20):
        [random.choice([0,1]) for i in range(l)]

    def get_good_strings(self,strings,lim=20):
        # print strings[0:5]
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

    def train_dA(self,data,corruption_level=0.2,num_epochs=200,lr=0.1):
        train_data = data
        # pdb.set_trace()
        train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
        sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=lr,num_epochs=num_epochs)

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

    def iterative_algorithm(
        self,
        name,
        pop_size=100,
        genome_length=20,
        lim_percentage=20,
        lim=20,
        trials=10,
        corruption_level=0.2,
        num_epochs=50,
        lr = 0.1,
        online_training=False,
        pickle_data=False,
        max_evaluations=200000):
        trials = max_evaluations/pop_size
        population_limit = lim
        if lim_percentage > 0:
            population_limit = pop_size/lim_percentage
        fitfile = open("{0}_fitnesses.dat".format(name),"w")
        self.dA = dA(n_visible=genome_length,n_hidden=50)
        self.dA.build_dA(corruption_level)
        self.build_sample_dA()
        all_strings,good_strings=self.generate_good_strings(pop_size,genome_length,population_limit)
        self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=50,lr=lr)
        sampled_population = [lo.sample_dA([i]) for i in self.get_new_population_rw(all_strings)]
        original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(all_strings,sampled_population)
        data = {
        "original":original_fitnesses,
        "fitnesses_sampled":sample_fitnesses,
        "differences_in_fitness":differences,
        "distances":distances
        }
        if pickle_data:
            pickle.dump(data,open("results/autoencoder/{0}_0.pkl".format(name),"wb"))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses),np.std(original_fitnesses)))
        fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
        print "{0},{1},{2}\n".format(np.mean(original_fitnesses),np.min(original_fitnesses),np.max(original_fitnesses))
        print "{0},{1},{2}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses))
        for iteration in range(0,trials):
            population = self.get_new_population_rw([z[0] for z in sampled_population])
            if online_training == False:
                self.dA = dA(n_visible=genome_length,n_hidden=50)
                self.dA.build_dA(corruption_level)
                self.build_sample_dA()
            good_strings,good_strings_fitnesses=self.get_good_strings(population,population_limit)
            self.train_dA(ar(good_strings),corruption_level=corruption_level,num_epochs=num_epochs,lr=lr)
            sampled_population = [lo.sample_dA([i]) for i in population]
            original_fitnesses,sample_fitnesses,differences,distances = self.get_statistics(population,sampled_population)
            data = {
            "original":original_fitnesses,
            "fitnesses_sampled":sample_fitnesses,
            "differences_in_fitness":differences,
            "distances":distances
            }
            if pickle_data:
                pickle.dump(data,open("results/autoencoder/{0}_{1}.pkl".format(name,iteration+1),"wb"))
            fitfile.write("{0},{1},{2},{3}\n".format(np.mean(sample_fitnesses),np.min(sample_fitnesses),np.max(sample_fitnesses),np.std(original_fitnesses)))
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
    args = sys.argv
    pop_size=int(args[1])
    genome_length=105
    lim_percentage=int(args[2])
    lim=int(args[3])
    trials=10
    corruption_level=0.0
    num_epochs=int(args[4])
    lr = float(args[5])
    online_training=int(args[6])
    if online_training == 0:
        online_training = False
    else:
        online_training = True
    pickle_data=False
    for i in range(0,10):
        for lim_percentage in [10,20,-20]:
            if lim_percentage < 0:
                lim_percentage = 0
                lim = 20
            name = "knapsack_105_items_{0}_{1}_{2}_{3}_{4}_{5}".format(pop_size,lim_percentage,num_epochs,lr,online_training,i)
            lo = ACSKnapsack(corruption_level=corruption_level,knapsack_file="weing8.pkl")
            # data =lo.experiment("c-{0}".format(c_level),no_trials=100,corruption_level=c_level)
            # t= Test()
            lo.iterative_algorithm(name,
                                    pop_size=pop_size,
                                    genome_length=genome_length,
                                    lim_percentage=lim_percentage,
                                    lim=lim,
                                    trials=10,
                                    corruption_level=corruption_level,
                                    num_epochs=num_epochs,
                                    lr = lr,
                                    online_training=online_training,
                                    pickle_data=False)
