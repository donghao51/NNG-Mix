from data_generator import DataGenerator
from myutils import Utils
import numpy as np
from baseline.DeepSAD.src.run import DeepSAD
import os
import argparse
from math import ceil
from myutils import Utils
from scipy import spatial
import pandas as pd
from baseline.Supervised import supervised

parser = argparse.ArgumentParser()
parser.add_argument('--la', type=float, default=0.1,
                    help='la')
parser.add_argument('--ratio', type=float, default=1.0,
                    help='ratio')
parser.add_argument('--mixup_alpha', type=float, default=0.2,
                    help='ratio')
parser.add_argument('--mixup_beta', type=float, default=0.2,
                    help='ratio')
parser.add_argument('--cutout_alpha', type=float, default=0.1,
                    help='ratio')
parser.add_argument('--cutout_beta', type=float, default=0.3,
                    help='ratio')
parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument("--method", type=str, default='nng_mix')
parser.add_argument("--alg", type=str, default='DeepSAD')
parser.add_argument("--dataset", type=str, default='Classical')
parser.add_argument('--use_anomaly_only', action='store_true')
parser.add_argument('--use_uniform', action='store_true')
parser.add_argument('--nn_k', type=int, default=10,
                    help='nn_k')
parser.add_argument('--nn_mix_gaussian', action='store_true')
parser.add_argument('--nn_mix_gaussian_std', type=float, default=1.0,
                    help='nn_mix_gaussian_std')
parser.add_argument('--adjust_nn_k', action='store_true')
parser.add_argument('--adjust_nn_k_n', type=int, default=2,
                    help='adjust_nn_k_n')
parser.add_argument("--appen", type=str, default='')
parser.add_argument('--gaussian_var', type=float, default=1.0,
                    help='gaussian_var')
parser.add_argument('--adjust_nn_k_anomaly', action='store_true')
parser.add_argument('--adjust_nn_k_n_anomaly', type=float, default=0.3,
                    help='adjust_nn_k_n_anomaly')
parser.add_argument('--nn_k_anomaly', type=int, default=10,
                    help='nn_k_anomaly')
args = parser.parse_args()

seed = args.seed
utils = Utils()
utils.set_seed(args.seed)

if args.dataset == 'Classical':
    dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets/Classical')
                                        if os.path.splitext(_)[1] == '.npz']
elif args.dataset == 'CV':
    dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets/CV_by_ResNet18')
                                        if os.path.splitext(_)[1] == '.npz']
else:
    dataset_list = [os.path.splitext(_)[0] for _ in os.listdir('datasets/NLP_by_BERT')
                                        if os.path.splitext(_)[1] == '.npz']

dataset_list.sort()

log_name = "log_%s_%s_%s_%s_%s"%(args.dataset, args.alg, args.method, str(args.la), str(args.ratio))
if args.method == 'mixup' or args.method == 'nng_mix':
    log_name = log_name + '_%s_%s'%(str(args.mixup_alpha), str(args.mixup_beta))
if args.method == 'cutout' or args.method == 'cutmix':
    log_name = log_name + '_%s_%s'%(str(args.cutout_alpha), str(args.cutout_beta))
if args.method == 'nng_mix' and (not args.adjust_nn_k):
    log_name = log_name + '_k_%s'%(str(args.nn_k))
if (args.method == 'nng_mix') and (not args.adjust_nn_k_anomaly):
    log_name = log_name + '_k_anomaly_%s'%(str(args.nn_k_anomaly))
if args.method == 'gaussian_noise':
    log_name = log_name + '_%s'%(str(args.gaussian_var))
if args.adjust_nn_k:
    log_name = log_name + '_adjust_nn_k_%s'%(str(args.adjust_nn_k_n))
if args.adjust_nn_k_anomaly:
    log_name = log_name + '_adjust_nn_k_anomaly_%s'%(str(args.adjust_nn_k_n_anomaly))
if args.nn_mix_gaussian:
    log_name = log_name + '_gaussian_std_%s'%(str(args.nn_mix_gaussian_std))
if args.use_uniform:
    log_name = log_name + '_use_uniform'
if args.use_anomaly_only:
    log_name = log_name + '_use_anomaly_only'

log_name = log_name + '_seed_%s'%(str(args.seed)) 

if args.appen:
    log_name = log_name + '_' + args.appen

log_name = log_name + '.csv'
base_path = "logs/"
log_path = base_path + log_name

num_times = [0, 1, 5, 10]

results = np.zeros((len(dataset_list), 4))

with open(log_path, "a") as f:
    f.write("{},{},{},{},{}\n".format('dataset', 0, 1, 5, 10))
    f.flush()
    da = 0
    for dataset in dataset_list:
        print(dataset)
        nu = 0
        for num in num_times:
            data_generator = DataGenerator(dataset=dataset, seed=args.seed)
            data, scaler = data_generator.generator(la=args.la, at_least_one_labeled=True) 

            anomaly_data = data['X_train'][data['y_train']==1]
            unlabeled_data = data['X_train'][data['y_train']==0]

            if args.ratio != 1.0:
                idx_choose_anomaly = np.random.choice(anomaly_data.shape[0], ceil(args.ratio * anomaly_data.shape[0]), replace=False)
                anomaly_data = anomaly_data[idx_choose_anomaly]

            # print(anomaly_data.shape)
            anomaly_data_copy = anomaly_data
            anomaly_data_g = np.empty([0, anomaly_data.shape[1]])
            gen_anomaly_files_n = anomaly_data.shape[0] * num

            print("anomaly_data.shape[0]: ", anomaly_data.shape[0])
            print("unlabeled_data.shape[0]: ", unlabeled_data.shape[0])

            if args.method == 'mixup':
                if args.use_anomaly_only:
                    for i in range(gen_anomaly_files_n):
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        index2 = np.random.choice(anomaly_data.shape[0], 1)
                        if anomaly_data.shape[0] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(anomaly_data.shape[0], 1)
                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)
                        anomaly_data_sample = (lam * anomaly_data[index2] + (1 - lam) * anomaly_data[index1])
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])
                else:
                    all_data = np.concatenate((anomaly_data, unlabeled_data), axis=0)
                    for i in range(gen_anomaly_files_n):
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        index2 = np.random.choice(all_data.shape[0], 1)
                        if anomaly_data.shape[0] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(all_data.shape[0], 1)
                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)
                        anomaly_data_sample = (lam * anomaly_data[index1] + (1 - lam) * all_data[index2])
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])

            elif args.method == 'cutout':
                dim = anomaly_data.shape[1]
                if args.use_anomaly_only:
                    for i in range(gen_anomaly_files_n):
                        cutoff_ratio = np.random.uniform(args.cutout_alpha,args.cutout_beta)
                        cutoff_num = int(cutoff_ratio*dim)
                        cutoff_num = max(1,cutoff_num)
                        
                        mask = np.ones((1, dim), np.float32)
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        x = np.random.choice(dim, 1)[0]
                        x1 = np.clip(x - cutoff_num // 2, 0, dim)
                        x2 = np.clip(x + cutoff_num // 2, 0, dim)
                        mask[:, x1: x2] = 0.

                        anomaly_data_sample = mask * anomaly_data[index1]
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])
                else:
                    all_data = np.concatenate((anomaly_data, unlabeled_data), axis=0)
                    for i in range(gen_anomaly_files_n):
                        cutoff_ratio = np.random.uniform(args.cutout_alpha,args.cutout_beta)
                        cutoff_num = int(cutoff_ratio*dim)
                        cutoff_num = max(1,cutoff_num)
                        
                        mask = np.ones((1, dim), np.float32)
                        index1 = np.random.choice(all_data.shape[0], 1)
                        x = np.random.choice(dim, 1)[0]
                        x1 = np.clip(x - cutoff_num // 2, 0, dim)
                        x2 = np.clip(x + cutoff_num // 2, 0, dim)
                        mask[:, x1: x2] = 0.

                        anomaly_data_sample = mask * all_data[index1]
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])

            elif args.method == 'cutmix':
                dim = anomaly_data.shape[1]
                if args.use_anomaly_only:
                    for i in range(gen_anomaly_files_n):
                        cutoff_ratio = np.random.uniform(args.cutout_alpha,args.cutout_beta)
                        cutoff_num = int(cutoff_ratio*dim)
                        cutoff_num = max(1,cutoff_num)
                        
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        index2 = np.random.choice(anomaly_data.shape[0], 1)
                        if anomaly_data.shape[0] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(anomaly_data.shape[0], 1)

                        x = np.random.choice(dim, 1)[0]
                        x1 = np.clip(x - cutoff_num // 2, 0, dim)
                        x2 = np.clip(x + cutoff_num // 2, 0, dim)
                        cutnum = x2 - x1
                        y = np.random.choice(dim-cutnum, 1)[0]
                        anomaly_data_sample = anomaly_data[index1]
                        anomaly_data_sample[:, y:y+cutnum] = anomaly_data[index2][:, x1:x2]
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])
                else:
                    all_data = np.concatenate((anomaly_data, unlabeled_data), axis=0)
                    for i in range(gen_anomaly_files_n):
                        cutoff_ratio = np.random.uniform(args.cutout_alpha,args.cutout_beta)
                        cutoff_num = int(cutoff_ratio*dim)
                        cutoff_num = max(1,cutoff_num)
                        
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        index2 = np.random.choice(all_data.shape[0], 1)
                        if anomaly_data.shape[0] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(all_data.shape[0], 1)

                        x = np.random.choice(dim, 1)[0]
                        x1 = np.clip(x - cutoff_num // 2, 0, dim)
                        x2 = np.clip(x + cutoff_num // 2, 0, dim)
                        cutnum = x2 - x1
                        y = np.random.choice(dim-cutnum, 1)[0]
                        anomaly_data_sample = anomaly_data[index1]
                        anomaly_data_sample[:, y:y+cutnum] = all_data[index2][:, x1:x2]
                        anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])

            elif args.method == 'gaussian_noise':
                dim = anomaly_data.shape[1]
                for i in range(gen_anomaly_files_n):
                    gaussian_noise = np.random.normal(0,args.gaussian_var,dim)
                    index1 = np.random.choice(anomaly_data.shape[0], 1)

                    anomaly_data_sample = gaussian_noise + anomaly_data[index1]
                    anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])

            elif args.method == 'nng_mix':
                dim = anomaly_data.shape[1]
                tree = spatial.KDTree(unlabeled_data)
                tree2 = spatial.KDTree(anomaly_data)
                for i in range(gen_anomaly_files_n):
                    if np.random.uniform(0, 1) > 0.5:
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        if args.adjust_nn_k:
                            dis, ind = tree.query(anomaly_data[index1], k=anomaly_data.shape[0]*args.adjust_nn_k_n)
                        else:
                            dis, ind = tree.query(anomaly_data[index1], k=args.nn_k)
                        index2 = np.random.choice(ind[0])

                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)
                        
                        if args.nn_mix_gaussian:
                            gaussian_noise1 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            gaussian_noise2 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            anomaly_data_sample = (lam * (gaussian_noise1 + anomaly_data[index1]) + (1 - lam) * (gaussian_noise2 + unlabeled_data[index2]))
                        else:
                            anomaly_data_sample = (lam * anomaly_data[index1] + (1 - lam) * unlabeled_data[index2])
                    else:
                        index1 = np.random.choice(anomaly_data.shape[0], 1)
                        if args.adjust_nn_k_anomaly:
                            query_k = int(anomaly_data.shape[0]*args.adjust_nn_k_n_anomaly)
                        else:
                            query_k = args.nn_k_anomaly
                        query_k = max(query_k, 1)
                        query_k = min(query_k, anomaly_data.shape[0])

                        dis, ind = tree2.query(anomaly_data[index1], k=query_k)
                    
                        ind = ind.reshape(1,-1)
                        if ind.shape[1] > 1:
                            index2 = np.random.choice(ind[0])
                        else:
                            index2 = ind[0]

                        if ind.shape[1] > 1:
                            while index2 == index1:
                                index2 = np.random.choice(ind[0])

                        if args.use_uniform:
                            lam = np.random.uniform(0, 1.0)
                        else:
                            lam = np.random.beta(args.mixup_alpha, args.mixup_beta)

                        if args.nn_mix_gaussian:
                            gaussian_noise1 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            gaussian_noise2 = np.random.normal(0,args.nn_mix_gaussian_std,dim)
                            anomaly_data_sample = (lam * (gaussian_noise1 + anomaly_data[index1]) + (1 - lam) * (gaussian_noise2 + anomaly_data[index2]))
                        else:
                            anomaly_data_sample = (lam * anomaly_data[index2] + (1 - lam) * anomaly_data[index1])
                    anomaly_data_g = np.vstack([anomaly_data_g, anomaly_data_sample])

            anomaly_data = np.vstack([anomaly_data_copy, anomaly_data_g])
            anomaly_labels = np.ones((anomaly_data.shape[0],1))

            X_train = np.vstack([unlabeled_data, anomaly_data])
            normal_labels = np.zeros((unlabeled_data.shape[0],1))
            
            y_train = np.concatenate(
                (normal_labels, anomaly_labels), axis=0)
            
            if args.alg == 'MLP':
                model = supervised(seed=args.seed, model_name='MLP')
            elif args.alg == 'DeepSAD':
                model = DeepSAD(seed=args.seed)
            
            model.fit(X_train=X_train, y_train=y_train[:,0])
            score = model.predict_score(data['X_test'])

            # evaluation
            utils = Utils()
            result = utils.metric(y_true=data['y_test'], y_score=score)

            results[da][nu] = result['aucroc']
            nu = nu + 1
        f.write("{},{},{},{},{}\n".format(dataset, results[da][0], results[da][1], results[da][2], results[da][3]))
        f.flush()
        da = da + 1
    result_mean = np.mean(results, axis=0)
    print(result_mean.shape)
    f.write("{},{},{},{},{}\n".format('avg', result_mean[0], result_mean[1], result_mean[2], result_mean[3]))
    f.flush()
f.close()