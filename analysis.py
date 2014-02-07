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
            max_fitnesses_new = np.loadtxt("{0}experiment_{1}/max_fitnesses.dat".format(path,i))
            mean_fitnesses_new = np.loadtxt("{0}experiment_{1}/mean_fitnesses.dat".format(path,i))
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
        return max_fitnesses,mean_fitnesses

    def analyse_random(self):
        path = "results/random/{0}/".format(self.experiment_name)
        max_fitnesses_new = np.loadtxt("{0}max_{1}".format(path,0))
        # return max_fitnesses_new
        max_fitnesses = np.ones((len(max_fitnesses_new),self.end))
        mean_fitnesses = np.ones((len(max_fitnesses_new),self.end))

        for i in range(self.start,self.end):
            path = "results/random/{0}/".format(self.experiment_name)
            max_fitnesses_new = np.loadtxt("{0}max_{1}".format(path,i))
            mean_fitnesses_new = np.loadtxt("{0}means_{1}".format(path,i))
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
        return max_fitnesses,mean_fitnesses

    def analyse_auto_encoder(self):
        path = "{0}".format(self.experiment_name)
        max_fitnesses_new = np.loadtxt("{0}_{1}_fitnesses.dat".format(path,0),delimiter=",")
        # return max_fitnesses_new
        max_fitnesses = np.ones((len(max_fitnesses_new),self.end))
        mean_fitnesses = np.ones((len(max_fitnesses_new),self.end))

        for i in range(self.start,self.end):
            # path = "results/random/{0}/".format(self.experiment_name)
            max_fitnesses_new = np.loadtxt("{0}_{1}_fitnesses.dat".format(path,i),delimiter=",").T[2].T
            mean_fitnesses_new = np.loadtxt("{0}_{1}_fitnesses.dat".format(path,i),delimiter=",").T[0].T
            if i == 0:
                max_fitnesses.T[0] = np.array([max_fitnesses_new]).T[2].T
                mean_fitnesses.T[0] = np.array([mean_fitnesses_new]).T[0].T
            else:
                max_fitnesses.T[i]=max_fitnesses_new.T[2].T
                mean_fitnesses.T[i] = mean_fitnesses_new.T[0].T
        plt.plot(np.mean(mean_fitnesses,axis=1))
        plt.savefig("{0}mean_fitnesses.png".format(path))
        plt.clf()
        plt.plot(np.mean(max_fitnesses,axis=1))
        plt.savefig("{0}max_fitnesses.png".format(path))
        return max_fitnesses,mean_fitnesses

if __name__ == "__main__":
    experiment_results = []
    # for exp,label in [  
    #                     ["knapsack_easy","ga_100"],
    #                     ["knapsack_easy_pop_size_500","ga_500"],
    #                     ["knapsack_easy_pop_size_1000","ga_1000"],
    #                     ["knapsack_easy_pop_size_10000","ga_10000"],
    #                 ]:
    #     e = Results(exp,no_runs=10,start=0,end=10)
    #     f,m=e.analyse()
    #     # plt.plot(np.mean(f,axis=1))
    #     mean_max = np.mean(f,axis=1)
    #     mean_mean = np.mean(f,axis=1)
    #     # plt.show()
    #     generations = len(m)
    #     x = range(0,100000+1,100000/(generations-1))
    #     plt.plot(x,mean_max[0:generations])
    #     experiment_results.append({
    #         "label":label,
    #         "mean_max":mean_max[0:generations],
    #         "mean_mean":mean_max[0:generations],
    #         "x_axis":x
    #         })
    # for exp,label in [  
    #                     ["easy_knapsack","random"],
    #                 ]:
    #     e = Results(exp,no_runs=10,start=0,end=10)
    #     f,m=e.analyse_random()
    #     mean_max = np.mean(f,axis=1)
    #     mean_mean = np.mean(f,axis=1)
    #     # plt.show()
    #     generations = len(m)
    #     x = range(0,100000+1,100000/(generations-1))
    #     # x = x[0:len(x)-1]
    #     # plt.plot(x,mean_max[0:generations])
    #     experiment_results.append({
    #         "label":label,
    #         "mean_max":mean_max[0:generations],
    #         "mean_mean":mean_max[0:generations],
    #         "x_axis":x[0:len(x)-1]
    #         })

    # plt.clf()
    # for e in experiment_results:
    #     plt.plot(e["x_axis"],e["mean_max"],label=e["label"])
    # plt.legend(loc="lower right")
    # plt.show()

    for exp,label in [  
                        ["test_knapsack","ae_top20_10000"],
                    ]:
        e = Results(exp,no_runs=10,start=0,end=10)
        f,m=e.analyse_auto_encoder()
        mean_max = np.mean(f,axis=1)[0:11]
        mean_mean = np.mean(f,axis=1)[0:11]
        # plt.show()
        generations = len(mean_max)
        x = range(0,100000+1,100000/(generations-1))
        # x = x[0:len(x)-1]
        # plt.plot(x,mean_max[0:generations])
        experiment_results.append({
            "label":label,
            "mean_max":mean_max[0:generations],
            "mean_mean":mean_max[0:generations],
            "x_axis":x[0:len(x)]
            })
    plt.clf()
    for e in experiment_results:
        plt.plot(e["x_axis"],e["mean_max"],label=e["label"])
    plt.legend(loc="lower right")
    plt.show()

