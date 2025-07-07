import random
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = {}
        for idx, item in enumerate(data_source.data):
            pid = item['id']
            self.index_dic.setdefault(pid, []).append(idx)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            self.length += len(idxs) - len(idxs) % self.num_instances

    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = self.index_dic[pid]
            random.shuffle(idxs)
            batch_idxs_dict[pid] = idxs
        avai_pids = self.pids.copy()
        final_idxs = []
        while len(avai_pids) >= 1:
            selected_pids = random.sample(avai_pids, 1)
            for pid in selected_pids:
                idxs = batch_idxs_dict[pid]
                if len(idxs) < self.num_instances:
                    avai_pids.remove(pid)
                    continue
                final_idxs.extend(idxs[:self.num_instances])
                batch_idxs_dict[pid] = idxs[self.num_instances:]
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length 