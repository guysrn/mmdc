import os
import time
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import numpy as np
import lap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.cluster.k_means_ as k_means
from sklearn.metrics.pairwise import euclidean_distances

from architectures.vgg import small_vgg
from architectures.resnet import resnet34, resnet18
from utils.metrics import accuracy, nmi_score
from data_handling.datasets import DatasetWithIndices, RotNetDataset
from utils.gmm import GaussianMixtureModel
from data_handling.transforms import train_transforms, inference_transforms


class MultiModalDeepClustering:
    def __init__(self, config, out_dir):
        self.config = config
        self.out_dir = out_dir
        self.train_dataloader, self.n_data = self._get_dataloader("train")
        self.eval_dataloader, _ = self._get_dataloader("eval")
        self.rot_dataloader, _ = self._get_dataloader("rot")
        self.gmm, self.targets = self._create_targets(self.n_data)
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        self.clustering_crit = nn.MSELoss()
        self.rot_crit = nn.CrossEntropyLoss()
        self._cuda()
        self.current_epoch = 0
        self.run_stats = {"acc": [], "nmi": [], "loss": []}

    def _cuda(self):
        self.model.cuda()
        self.clustering_crit.cuda()
        self.rot_crit.cuda()

    def load_checkpoint(self, path):
        checkpoint = torch.load(os.path.join(path, "model.pt"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.targets = checkpoint["targets"]
        self.out_dir = path
        with open(os.path.join(path, "run_stats.pickle"), "rb") as handle:
            self.run_stats = pickle.load(handle)

    def save_checkpoint(self, path):
        torch.save({"epoch": self.current_epoch,
                    "targets": self.targets,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict()},
                     os.path.join(path, "model.pt"))

        with open(os.path.join(path, "run_stats.pickle"), "wb") as handle:
            pickle.dump(self.run_stats, handle)

    def train(self):
        print(f"{datetime.now()} Pre-training evaluation:")

        loss, nmi, acc = self._evaluation()
        print(f"loss: {loss}, acc: {acc}, nmi: {nmi}")

        for e in range(self.current_epoch, self.config.epochs):
            print(f"\n{datetime.now()} epoch {e}/{self.config.epochs}")
            end = time.time()

            if self.config.refine_epoch == e:
                print(f"{datetime.now()} starting refinement stage, targets will be reassigned using k-means")
                with open(os.path.join(self.out_dir, "no_refine_run_stats.pickle"), "wb") as handle:
                    pickle.dump(self.run_stats, handle)

            if self.config.refine_epoch <= e:
                # we are in refinement stage, reassign targets with k-means
                preds = []
                self.model.eval()

                for batch in self.eval_dataloader:
                    images, _ = batch
                    preds.append(self.model(images.cuda()).data.cpu().numpy())

                preds = np.concatenate(preds)
                _, labels, _ = k_means.k_means(preds, self.config.k)

                # find permutation of labels that is closest to previous
                num_correct = np.zeros((self.config.k, self.config.k))
                prev_labels = np.argmax(self.targets, axis=1)
                for c_1 in range(self.config.k):
                    for c_2 in range(self.config.k):
                        num_correct[c_1, c_2] = int(((labels == c_1) * (prev_labels == c_2)).sum())
                _, assignments, _ = lap.lapjv(self.n_data - num_correct)
                reordered = np.zeros(self.n_data, dtype=np.int)
                for c in range(self.config.k):
                    reordered[labels == c] = assignments[c]

                self.targets = np.eye(self.config.k)[reordered]

            if self.config.rotnet:
                # train an epoch on rotation auxiliary task
                for batch in self.rot_dataloader:
                    images, labels = batch

                    unpack_images = []
                    for i in range(len(images[0])):
                        for r in range(4):
                            unpack_images.append(images[r][i])

                    unpack_images = np.stack(unpack_images, axis=0)
                    labels = np.reshape(labels, newshape=-1)

                    self.model.train()

                    images = torch.tensor(unpack_images, dtype=torch.float, device="cuda")
                    labels = labels.cuda()

                    out = self.model(images, rot_head=True)
                    loss = self.rot_crit(out, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # train an epoch on main clustering task
            for batch in self.train_dataloader:
                images1, images2, indices = batch

                if self.config.refine_epoch > e:
                    # optimize and update targets
                    self.model.eval()

                    pred = self.model(images1.cuda()).data.cpu().numpy()

                    batch_targets = self.targets[indices]

                    cost = euclidean_distances(pred, batch_targets)
                    _, assignments, _ = lap.lapjv(cost)

                    for i, idx in enumerate(indices):
                        self.targets[idx] = batch_targets[assignments[i]]

                images = images2.cuda()
                batch_targets = torch.tensor(self.targets[indices], dtype=torch.float, device="cuda")

                self.model.train()
                pred = self.model(images)
                loss = self.clustering_crit(pred, batch_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()

            loss, nmi, acc = self._evaluation()
            self.run_stats["loss"].append(loss)
            self.run_stats["acc"].append(acc)
            self.run_stats["nmi"].append(nmi)

            print(f"{datetime.now()} train epoch took: {int(time.time() - end)}s")
            print(f"{datetime.now()} loss: {loss}, acc: {acc}, nmi: {nmi}")

            self.current_epoch = e

            if e % self.config.plot_rate == 0:
                fig, ax = plt.subplots(len(self.run_stats), figsize=(10, 30))

                for i, run_stat_name in enumerate(self.run_stats.keys()):
                    ax[i].plot(range(e + 1), self.run_stats[run_stat_name])
                    title = run_stat_name + ' (' + str(format(self.run_stats[run_stat_name][-1], '.4f')) + ')'
                    ax[i].set_title(title)

                plt.savefig(os.path.join(self.out_dir, "plots"))
                plt.close()

                self.save_checkpoint(self.out_dir)

    def _evaluation(self):
        pred = []
        labels = []

        self.model.eval()

        for batch in self.eval_dataloader:
            images, targets = batch
            pred.append(self.model(images.cuda()).data.cpu().numpy())
            labels.append(targets)

        pred = np.concatenate(pred, axis=0)
        labels = np.concatenate(labels, axis=0)
        pred_labels = self.gmm.get_labels(pred)

        nmi = nmi_score(pred_labels, labels)
        acc, _, _ = accuracy(pred_labels, labels, self.config.k, len(pred_labels))
        loss = self.clustering_crit(torch.tensor(pred, dtype=torch.float, device="cuda"),
                                    torch.tensor(self.targets, dtype=torch.float, device="cuda")).item()

        return loss, nmi, acc

    def _get_optimizer(self):
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1:  # not adding weight decay to batchnorm params
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': self.config.wd}]
        return SGD(params=params, lr=self.config.lr, momentum=self.config.momentum)

    def _get_lr_scheduler(self, optimizer):
         return lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=self.config.lr_decay_epochs,
                                         gamma=self.config.lr_decay_gamma)

    def _get_model(self):
        in_channels = 1 if self.config.dataset == "mnist" else 3
        out_dim = self.config.gmm_dim if self.config.gmm_dim else self.config.k
        if self.config.arch == "vgg":
            model = small_vgg(in_size=self.config.input_size,
                              in_channels=in_channels,
                              sobel=self.config.sobel,
                              out_dim=out_dim)
        elif self.config.arch == "resnet34":
            model = resnet34(in_channels=in_channels,
                             rotnet_head=self.config.rotnet,
                             sobel=self.config.sobel,
                             out_dim=out_dim)
        elif self.config.arch == 'resnet18':
            model = resnet18(in_channels=in_channels,
                             rotnet_head=self.config.rotnet,
                             sobel=self.config.sobel,
                             out_dim=out_dim)
        else:
            raise ValueError(f"Unknown arch: {self.config.arch}")

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"gpu count is {n_gpus}, using DataParallel")
            model = nn.DataParallel(model)

        return model

    def _create_targets(self, n_data):
        gmm_dim = self.config.gmm_dim if self.config.gmm_dim else self.config.k
        gmm = GaussianMixtureModel(sigma=self.config.sigma, k=self.config.k, dim=gmm_dim, init=self.config.gmm_means)
        targets = gmm.sample(n_data)
        np.random.shuffle(targets)
        return gmm, targets

    def _get_dataloader(self, mode):
        if mode == "train":
            transform1 = inference_transforms(crop_size=self.config.crop_size[0],
                                             input_size=self.config.input_size)
            transform2 = train_transforms(crop_sizes=self.config.crop_size,
                                         input_size=self.config.input_size,
                                         flip=self.config.flip,
                                         color_jitter=self.config.color_jitter,
                                         rot_degree=self.config.rot_degree)
            dataset = DatasetWithIndices(root=self.config.root,
                                         dataset=self.config.dataset,
                                         transform1=transform1,
                                         transform2=transform2,
                                         train=True)
        elif mode == "eval":
            transform = inference_transforms(crop_size=self.config.crop_size[0],
                                             input_size=self.config.input_size)
            dataset = DatasetWithIndices(root=self.config.root,
                                         dataset=self.config.dataset,
                                         transform1=transform,
                                         transform2=None,
                                         train=False)
        elif mode == "rot":
            if self.config.rotnet:
                transform = train_transforms(crop_sizes=self.config.crop_size,
                                             input_size=self.config.input_size,
                                             flip=False,
                                             color_jitter=False,
                                             rot_degree=0)
                dataset = RotNetDataset(root=self.config.root, dataset=self.config.dataset, transform=transform)
            else:
                return None, None
        else:
            raise ValueError(f"Unknown dataloader mode: {mode}")

        return DataLoader(dataset=dataset,
                          batch_size=self.config.batch_size,
                          shuffle=not mode == "eval",
                          num_workers=self.config.workers,
                          pin_memory=True), len(dataset)
