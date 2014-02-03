import numpy as np
import matplotlib.pylab as plt

class Results(object):
    def __init__(self, experiment_name,no_runs=5,start=0,end=None):
        super(Results, self).__init__()
        self.experiment_name = experiment_name
        self.no_runs = no_runs
        self.start = start
        self.end = end
        if self.end == None:
            self.end = self.no_runs

    def analyse(self):
        path = "results/{0}/".format(self.experiment_name)
        max_fitnesses_new = np.loadtxt("{0}experiment_{1}/max_fitnesses.dat".format(path,0))
        # return max_fitnesses_new
        max_fitnesses = np.ones((len(max_fitnesses_new),self.end))
        mean_fitnesses = np.ones((len(max_fitnesses_new),self.end))

        for i in range(self.start,self.end):
            path = "results/{0}/".format(self.experiment_name)
            max_fitnesses_new = np.loadtxt("{0}experiment_{1}/max_fitnesses.dat".format(path,i))[0:11]
            mean_fitnesses_new = np.loadtxt("{0}experiment_{1}/mean_fitnesses.dat".format(path,i))[0:11]
            if i == 0:
                max_fitnesses.T[0] = np.array([max_fitnesses_new])
                mean_fitnesses.T[0] = np.array([mean_fitnesses_new])
            else:
                max_fitnesses.T[i]=max_fitnesses_new
                mean_fitnesses.T[i] = mean_fitnesses_new
        plt.plot(np.mean(mean_fitnesses,axis=1))
        plt.savefig("{0}mean_fitnesses.png".format(path))
        plt.clf()
        plt.plot(np.mean(max_fitnesses,axis=1))
        plt.savefig("{0}max_fitnesses.png".format(path))
        return mean_fitnesses

if __name__ == "__main__":
    e = Results("one_max",no_runs=5,start=0,end=5)
    f=e.analyse()
    # plt.plot(np.mean(f,axis=1))
    # plt.show()
