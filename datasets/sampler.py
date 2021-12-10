from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        #print(len(final_idxs))
        #print(len(np.unique(final_idxs)))
        return iter(final_idxs)

    def __len__(self):
        return self.length


class ClusterIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        self.cluster_dic = defaultdict(list)  # dict with list value
        self.cluster_ids = defaultdict(list)

        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, cluster_id) in enumerate(self.data_source):
            self.index_dic[pid].append(index)

            self.cluster_ids[index] = cluster_id
            self.cluster_dic[str(pid)+'_'+str(cluster_id)].append(index)
            # for each ID

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):

        ## CLUSTER_NUM
        CLUSTER_NUM =  10


        random.shuffle(self.pids)

        batch_idxs_dict = []#list()
        for cluster_select in range(CLUSTER_NUM):
            for pid in self.pids:
            #cluster_select = np.random.randint(CLUSTER_NUM,size=1)[0]
                #print(str(pid) + '_' + str(cluster_select))
                #print(self.cluster_dic[str(pid) + '_' + str(cluster_select)])
                idxs = copy.deepcopy(self.cluster_dic[str(pid) + '_' + str(cluster_select)])

                #idxs = copy.deepcopy(self.index_dic[pid])

                if len(idxs) == 0:
                    continue
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict.append(batch_idxs)
                        batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []


        while len(batch_idxs_dict) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            # add cluster regularization

            if len(batch_idxs_dict) == 0:
                break
                avai_pids.remove(pid)


            for _ in range(self.num_pids_per_batch):
                batch_idxs = batch_idxs_dict.pop(0)

                #print(batch_idxs)
                final_idxs.extend(batch_idxs)

        #print(len(np.unique(final_idxs)))
        #print(len(final_idxs))
        #print(final_idxs)
        #print (self.cluster_ids[np.array(final_idxs[0])])
        return iter(final_idxs)

    def __len__(self):
        return self.length