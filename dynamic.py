# %%
import numpy as np
from scipy.spatial.distance import cdist
import gzip
import os
import bisect
import multiprocessing
from math import floor,ceil
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import tarfile
import coresets
import pickle
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import traceback
import concurrent.futures
import pandas as pd
import logging
import sys
import uuid
import cProfile
import pstats
import traceback
import shutil
from concurrent.futures import ProcessPoolExecutor
test = 0
# %%
def load_data(dataset_name):
    Verify_X = None
    Verify_labels = None
    if dataset_name == "MNIST":
        path = '/home/sjj/Experiment/data/MNIST/raw'
        labels_path = os.path.join(path, f'train-labels-idx1-ubyte.gz')
        images_path = os.path.join(path, f'train-images-idx3-ubyte.gz')

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            X = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    elif dataset_name == "Cifar-10":
        path = '/home/sjj/Experiment/data/cifar-10-python.tar.gz'
        with tarfile.open(path) as tar:
            tar.extractall()
            X = []
            labels = []
            for i in range(1, 6):
                batch_dict = unpickle(f'cifar-10-batches-py/data_batch_{i}')
                X.append(batch_dict[b'data'])
                labels.append(batch_dict[b'labels'])
            X = np.concatenate(X)
            labels = np.concatenate(labels)
    elif dataset_name == "Covertype":
        path="/home/sjj/Experiment/data/covtype.data"
        df_cov = pd.read_csv(path, header=None)
        df_np = df_cov.to_numpy()
        X = df_np[:,:-1]
        labels = df_np[:,-1]
    elif dataset_name == "USCensus":
        path="/home/sjj/Experiment/data/USCensus1990.data.txt"
        df_census = pd.read_csv(path, header=1)
        # 将DataFrame转换为NumPy数组
        np_array = df_census.to_numpy()
        X = np_array[:, 1:]
        labels = np_array[:, 0]
    elif dataset_name == "modified_covtype":
        path = "/home/sjj/Experiment/data/covtype.data"
        
        df_cov = pd.read_csv(path, header=None)
        # 将数据转换为 numpy 数组
        df_np = df_cov.to_numpy(dtype=np.float64)
        # 提取特征和标签
        X_ = df_np[:, :-1]
        #labels_ = df_np[:, -1]
        origin_labels = df_np[:, -1]
        # 将标签为 2 的值改为 1，其余标签置为 0
        labels_ = np.where(origin_labels == 2, 1, 0)
        #labels = labels.flatten()
        ratio = 0.9
        if test:
            ratio = 0.1

        X_train_indices = np.random.choice(X_.shape[0], int(X_.shape[0] * ratio), replace=False)
        X_test_indices = np.setdiff1d(np.arange(X_.shape[0]), X_train_indices)

        X = X_[X_train_indices]
        labels = labels_[X_train_indices].flatten()

        Verify_X = X_[X_test_indices]
        Verify_labels = labels_[X_test_indices].flatten()
        

    elif dataset_name == "Tripfare":
        path = "/home/sjj/Experiment/data/taxi_fare/train.csv"
        data = pd.read_csv(path)
        #data
        data_np = data.to_numpy()
        if test:
            labels = data_np[:50000,-2]
            X = np.delete(data_np, -2, axis=1)[:50000,:]
        else:
            labels = data_np[:,-2]
            X = np.delete(data_np, -2, axis=1)
    elif dataset_name == "Energy":
        path = "/home/sjj/Experiment/data/KAG_energydata_complete.csv"
        data = pd.read_csv(path)
        data_np = data.to_numpy()

        X = data_np[:,2:]
        labels = data_np[:,1]
    elif dataset_name == "Query":
        path =  "/home/sjj/Experiment/data/Datasets/Range-Queries-Aggregates.csv"
        data = pd.read_csv(path)
        data = data.dropna(how="any")
        data_np = data.to_numpy()
        X = data_np[:,1:5]
        labels = data_np[:,7]
    elif dataset_name == "GPU":
        path = "/home/sjj/Experiment/data/sgemm_product.csv"
        data = pd.read_csv(path)
        X = data.to_numpy()[:,:14]
        labels = data.to_numpy()[:,14:].mean(axis=1)
    elif dataset_name == "HAR":
        path = '/home/sjj/Experiment/data/aggregated_data_with_new_label.csv'
        data = pd.read_csv(path)
        data = data.dropna()
        X_ = data.to_numpy()[:,:-1]
        labels_ = data.to_numpy()[:,-1]
        
        X_train_indices = np.random.choice(X_.shape[0], int(X_.shape[0]*0.9), replace=False)
        X_test_indices = np.setdiff1d(np.arange(X_.shape[0]), X_train_indices)

        X = X_[X_train_indices]
        labels = labels_[X_train_indices].flatten()

        Verify_X = X_[X_test_indices]
        Verify_labels = labels_[X_test_indices].flatten()
    elif dataset_name == "KDD":
        path = "/home/sjj/Experiment/data/kddcup99/kddcup_10_percent.csv"
        data = pd.read_csv(path)
        data = data.dropna()
        X_ = data.to_numpy()[:,:-1]
        labels_ = data.to_numpy()[:,-1]

        X_train_indices = np.random.choice(X_.shape[0], int(X_.shape[0]*0.9), replace=False)
        X_test_indices = np.setdiff1d(np.arange(X_.shape[0]), X_train_indices)

        X = X_[X_train_indices]
        labels = labels_[X_train_indices].flatten()

        Verify_X = X_[X_test_indices]
        Verify_labels = labels_[X_test_indices].flatten()
    return X,labels,Verify_X,Verify_labels
# %%
def check(node):
    while node is not None:
        for item in node.data_instances:
            if (type(item) != list)|(len(item)!=55):
                print("False")
                return False
            check(node.left_child)
            check(node.right_child)
    return True
def get_uuid():
    return str(uuid.uuid4())

# %%
## here is a dynamic implement
def logistic_loss(X,y,problem_attributes=None,theta=None,weights=None):
    # if (type(X) == list) &(type(X[0]) != list):
    #     X = np.array(X).reshape(1,-1)
    #     y = np.array(y)
    try:
        if weights is None:
            weights = np.ones(X.shape[0])
        if theta is None:
            initial_model = LogisticRegression(max_iter=1)
            initial_model.fit(X,y,weights)
            theta = np.insert(initial_model.coef_,0,initial_model.intercept_)
        intercept = theta[0]
        w = theta[1:]
        np_w = np.array(w).reshape(-1,1)
        losses = np.empty(X.shape[0])
        # for i in range(X.shape[0]):
        #     x_i_np = X[i].reshape(-1,1)
        #     tmp = weights[i]/np.sum(weights)*(-y[i]*np.log(1/(1+np.exp(-intercept-np.dot(np_w,x_i_np))))-(1-y[i])*np.log(1-1/(1+np.exp(-intercept-np.dot(np_w,x_i_np)))))
        #     losses[i] = tmp[0][0]
        # return losses

        linear_combination = intercept + np.dot(X, np_w)

        # 计算预测概率
        tol = 1e-10
        
        predictions = 1 / (1 + np.exp(-linear_combination))
        #print(predictions)
        # 确保预测值在合理范围内
        predictions = np.clip(predictions, tol, 1 - tol)
        #print(predictions)
        log_preds = np.log(predictions+tol)
        log_one_minus_preds = np.log(1 - predictions+tol)
        #print((weights/np.sum(weights)).shape)
        losses = -(weights).reshape(-1,1) * (y.reshape(-1,1) * log_preds + (1 - y.reshape(-1,1)) * log_one_minus_preds)
        #print(log_one_minus_preds.shape)
        return losses.flatten()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None

def H(z,epsilon=1.35):
    ans = np.where(
        np.abs(z) < epsilon,
        (z**2),
        (2 * epsilon * np.abs(z) - epsilon**2)
    )
    return ans.flatten()
def huber_regression_loss(X,y,z=None,problem_attributes=1.35,theta=None,weights=None):
    if (type(X) == list) &(type(X[0]) != list):
        X = np.array(X).reshape(1,-1)
    else:
        X = np.array(X)
    y = np.array(y)
    try:
        if problem_attributes is None:
            problem_attributes = 1.35
        if weights is None:
            weights = np.ones(X.shape[0])
        if theta is None:
            initial_model = HuberRegressor(max_iter=1,alpha=0)
            #print("?")
            initial_model.fit(X,y,weights)
            #print("?")
            theta = np.insert(initial_model.coef_,0,[initial_model.scale_,initial_model.intercept_])
        sigma = theta[0]
        intercept = theta[1]
        w = theta[2:]
        np_w = np.array(w).reshape(-1,1)
        #w = np.array(w).reshape(1,-1)

        epsilon = problem_attributes
        unweighted_loss = (sigma+H((y.reshape(-1,1)-intercept-np.dot(X,np_w))/sigma,epsilon)*sigma).reshape(-1,1)
        if z is None:
            losses = weights.reshape(-1,1) * unweighted_loss

        if z is not None:
            #print(z)
            sorted_indices = np.argsort(unweighted_loss.flatten())
            cumulative_weights = np.cumsum(weights[sorted_indices])
            normal_indices = sorted_indices[np.where(cumulative_weights <= cumulative_weights[-1]-z)[0]]
            outlier_indices = np.setdiff1d(sorted_indices, normal_indices)

            outlier_weight = np.sum(weights[outlier_indices])
            weights_ = weights
            if normal_indices.size > 0: 
                normal_max = np.argmax(unweighted_loss[normal_indices])
                weights_[normal_max] = weights[normal_max]-z+outlier_weight

            losses = (weights_[normal_indices].reshape(-1,1))*(unweighted_loss[normal_indices].reshape(-1,1))

        # losses = np.empty(X.shape[0])
        # for i in range(X.shape[0]):
        #     x_i_np = X[i].reshape(-1,1)
        #     tmp = weights[i]*(sigma + H((y[i]-intercept-np.dot(np_w,x_i_np))/sigma,epsilon))
        #     losses[i] = tmp
    except Exception as e:
        print(e)
        print("Error")
    return losses.flatten()
# %%
def kmeans_loss(X,y = None,problem_attributes=None,theta=None,weights=None):
    
    if theta is None:
        initial_model = KMeans(n_clusters=problem_attributes, init='k-means++', n_init=1, max_iter=1, random_state=0)
        initial_model.fit(X)
        theta = initial_model.cluster_centers_
    if weights is None:
        weights = np.ones(X.shape[0])
    #print(X.shape,theta.shape)
    labels = np.argmin(cdist(X, theta), axis=1)
    #losses = np.empty(X.shape[0])
    #print((X - theta[labels])**2.shape)
    losses = weights.reshape(-1,1)*(np.sum((X - theta[labels])**2,axis=1)).reshape(-1,1)
    #for i in range(X.shape[0]):
    #    losses[i] = weights[i]*np.sum((X[i] - theta[labels[i]]) ** 2)
    return losses.flatten()
# %%
def coreset_ring(X,y=None,weights = None, problem_attributes=None, coreset_size=1000, 
                 initial_centers=None, loss_function=None, thresh_hold = None):
    try:
        if loss_function != kmeans_loss:
            print("Only kmeans supported")
            return
        losses = loss_function(X,y,problem_attributes=problem_attributes,theta = initial_centers)
        max = losses.max()
        min = losses.min()
        # parameter? or out
        n_samples, n_features = X.shape
        
        layer = np.log2(max/min).astype(int)+1
        if thresh_hold is None:
            thresh_hold = (n_samples/layer)*losses.mean()/50
        # indices_layer = [[] for _ in range(layer)]
        
        group_losses = [0]*layer
        group_indices = [[] for _ in range(layer)]
        group_count = 0

        ring_losses = [0]*layer
        ring_indices = [[] for _ in range(layer)]
        ring_count = 0
        lower = min
        upper = 2*min
        group_session = False
        if test:
            print(thresh_hold)
            print(X.shape,y.shape)
            # print(lower,upper)
        for _ in range(layer):
            indices = np.where((losses >= lower) & (losses < upper))[0]
            if len(indices)>0:      
                losses_sum = losses[indices].sum()
                if test:
                    print(_,losses_sum,len(indices))
                if  losses_sum > thresh_hold:
                    ring_indices[ring_count].extend(indices)
                    ring_losses[ring_count] = losses_sum
                    ring_count +=1
                    if group_session:
                        group_session = False
                        group_count +=1
                else:
                    if group_session:
                        if losses_sum + group_losses[group_count] > thresh_hold:
                            group_session = False
                            group_count += 1
                            group_session = True
                            group_indices[group_count].extend(indices)
                            group_losses[group_count] += losses_sum
                        else:
                            group_indices[group_count].extend(indices)
                            group_losses[group_count] += losses_sum
                    else:
                        group_session = True
                        group_indices[group_count].extend(indices)
                        group_losses[group_count] += losses_sum
            upper *=2
            lower *=2
       
        indices = []
        samples_per_layer = int((coreset_size-2*group_count)/ring_count)
        samples_layer = [0]*ring_count
        not_full = range(ring_count) 
        if test:
            print("before")
            print(ring_losses)
            print([len(item) for item in ring_indices])
            print(group_losses)
            print([len(item) for item in group_indices])
            print(coreset_size)
            print(samples_per_layer)
           
        while True:
            new_full = []
            new_not_full = []
            size_coreset_i = 0

            for i in not_full:
                
                if len(ring_indices[i]) -samples_layer[i]< samples_per_layer:
                    size_coreset_i += len(ring_indices[i]) - samples_layer[i]
                    samples_layer[i] = len(ring_indices[i])
                    new_full.append(i)
                else:
                    size_coreset_i += int(samples_per_layer)
                    samples_layer[i] += int(samples_per_layer)
                    new_not_full.append(i)
            if not new_full:
                break
            else:
                if test:
                    print(new_not_full,new_full)
                if len(new_not_full) ==0:
                    break
                samples_per_layer = int((coreset_size - 2*group_count - size_coreset_i)/len(new_not_full))
                if test:
                    print(samples_per_layer)
                not_full = new_not_full
        if test:
            print(samples_layer)
        coreset = np.array([]).reshape(0, n_features + 1)
        coreset_y = np.array([]).reshape(0,1)
        tmp = np.array([]).reshape(0, n_features + 1)
        tmp_y = np.array([]).reshape(0,1)
        # ring
        final_indices = []
        for i in range(ring_count):
            if samples_layer[i] > 0:
                weight_i = len(ring_indices[i])/samples_layer[i]
                if samples_layer[i] == len(ring_indices[i]):
                    X_i = np.hstack((X[ring_indices[i]],np.full((len(ring_indices[i]),1),weight_i)))
                    y_i = y[ring_indices[i]].reshape(-1,1)
                    final_indices.extend(ring_indices[i])
                else:
                    coreset_indices = np.random.choice(len(ring_indices[i]),samples_layer[i])
                    original_indices = np.array(ring_indices[i])[coreset_indices]
                    X_i = np.hstack((X[original_indices],np.full((samples_layer[i],1),weight_i)))
                    y_i = y[original_indices].reshape(-1,1)
                    final_indices.extend(original_indices)
                tmp= np.vstack((tmp, X_i))
                # print(tmp_y.shape,y_i.shape)
                tmp_y = np.vstack((tmp_y,y_i))
        coreset = np.vstack((coreset, tmp))
        coreset_y = np.vstack((coreset_y, tmp_y))
        # group
        if group_count != 0:
            tmp = np.array([]).reshape(0, n_features + 1)
            tmp_y = np.array([]).reshape(0,1)
            for i in range(group_count):
                if len(group_indices[i]) <= 2:
                    X_new = np.hstack((X[group_indices[i]],np.full((len(group_indices[i]),1),1)))
                    y_new = y[group_indices[i]].reshape(-1,1)
                    final_indices.extend(group_indices[i])
                    tmp = np.vstack((tmp,X_new))
                    tmp_y = np.vstack((tmp_y,y_new))
                    continue
                p_far_ind = np.argmax(losses[group_indices[i]])
                p_close_ind = np.argmin(losses[group_indices[i]])
                X_far_ind = group_indices[i][p_far_ind]
                X_close_ind = group_indices[i][p_close_ind]
                final_indices.extend([X_far_ind,X_close_ind])
                p_far = X[X_far_ind].reshape(1,-1)
                # print(p_far.shape)
                p_close = X[X_close_ind].reshape(1,-1)
                loss_far = losses[X_far_ind]
                loss_close = losses[X_close_ind]
                
                cur_loss = losses[group_indices[i]]
                if test:
                    print(loss_close,loss_far)
                weight_close = ((cur_loss - loss_far)/(loss_close - loss_far)).sum()
                weight_far = len(group_indices[i]) -weight_close
                # print(weight_close)
                X_far = np.hstack((p_far,np.full((1,1),weight_far)))
                X_close = np.hstack((p_close,np.full((1,1),weight_close)))
                X_new = np.vstack((X_far, X_close))

                y_new = np.vstack((y[X_far_ind],y[X_close_ind])).reshape(-1,1)
                tmp = np.vstack((tmp,X_new))
                tmp_y = np.vstack((tmp_y,y_new))
            coreset = np.vstack((coreset, tmp))
            coreset_y = np.vstack((coreset_y, tmp_y))
        if weights is not None:
            coreset[:,-1] = coreset[:,-1]*weights[final_indices]
        return coreset, coreset_y.reshape(-1,1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None


# %%
# generalized sampling method to weighted data
def coreset_uniform(X, y=None, weights = None, coreset_size=1000):
    
    #print(len(X))
    # try:
    #     X_copy = np.array(X.copy())
    # except:
    #     # for item in X:
    #     #     print(len(item))
    #     traceback.print_exc()
    #     print(X)
    #     #return
    # #print(X_copy.shape)
    # #print(X)
    # try:
    #     y = np.array(y).flatten()   
        
    # except:
    #     traceback.print_exc()
    #     print(y)
    #print(len(X),len(y))
    n_samples, n_features = X.shape
    indices = np.random.choice(n_samples, size=coreset_size, replace=False)
    sample_weights = np.ones((n_samples,)) * n_samples / coreset_size
    
    coreset = np.hstack((X[indices], sample_weights[indices].reshape(-1, 1)))
    if weights is not None:
        weights = np.array(weights)
        coreset[:,-1] = coreset[:,-1]*weights[indices]
    return coreset, y[indices].reshape(-1,1)
# %%
# generalized sampling method to weighted data
def coreset_GSP(X, y=None, weights=None,problem_attributes=None, coreset_size = 1000, initial_centers  = None, loss_function = kmeans_loss):
    try:
       
        
        n_samples, n_features = X.shape
        #centers = initial_centers
        losses = loss_function(X,y, problem_attributes=problem_attributes, theta=initial_centers)
        #print(losses)
        
        T = losses.mean()
        tau = losses.min()
        lower = tau
        upper = T + tau
        layer = 0
        # 这里label指第几层
        X_with_labels = []
        Y_with_labels = []
        #w_with_labels = []
        # 这里没维护indices 有点冗余了
        for _ in range(np.log2(n_samples).astype(int) + 1):
            indices = np.where((losses >= lower) & (losses < upper))[0]
            if len(indices) > 0 :
                label = np.full((len(indices),), layer)
                X_with_labels.append(np.hstack((X[indices], label.reshape(-1, 1))))
                #print(X_copy[indices].shape)
                #print(y[indices].reshape(-1,1).shape)
                #print(label.reshape(-1,1).shape)
                Y_with_labels.append(np.hstack((y[indices].reshape(-1,1),label.reshape(-1,1))))
                #w_with_labels.append(np.hstack((weights[indices].reshape(-1,1),label.reshape(-1,1))))
                layer += 1
            lower = tau + T * np.power(2,_)
            upper = tau + T * np.power(2,_+1)
        samples_per_layer = coreset_size / layer
        X_with_labels = np.vstack(X_with_labels)
        Y_with_labels = np.vstack(Y_with_labels)
        #w_with_labels = np.vstack(w_with_labels)
        
        coreset = np.array([]).reshape(0, n_features + 1)
        coreset_y = np.array([]).reshape(0, 2)
        #coreset_w = np.array([]).reshape(0,2)
        full = []
        indices = []
        not_full = range(layer)
        while True:
            new_full = []
            new_not_full = []
            #..... 写错了
            for i in not_full:
                X_i = X_with_labels[X_with_labels[:, -1] == i,:]
                Y_i = Y_with_labels[Y_with_labels[:, -1] == i,:]
                #w_i = w_with_labels[w_with_labels[:, -1] == i,:]
                if len(X_i) < samples_per_layer:
                    new_full.append(i)
                    indices.extend(np.where(X_with_labels[:, -1] == i)[0])
                    coreset = np.vstack((coreset, X_i))
                    coreset_y = np.vstack((coreset_y,Y_i))
                    #coreset_w = np.vstack((coreset_w,w_i))
                else:
                    new_not_full.append(i)
                    coreset_indices = np.random.choice(X_i.shape[0], int(samples_per_layer))
                    original_indices = (np.where(X_with_labels[:, -1] == i)[0])[coreset_indices]
                    indices.extend(original_indices)
                    coreset = np.vstack((coreset, X_i[coreset_indices]))
                    coreset_y = np.vstack((coreset_y,Y_i[coreset_indices]))
                    #coreset_w = np.vstack((coreset_w,w_i[coreset_indices]))
            if not new_full:
                break
            else:
                samples_per_layer = (coreset_size - len(coreset)) / len(new_not_full)
                not_full = new_not_full
                full = new_full

        
        # weight 
        for i in range(layer):
            count_X = np.count_nonzero(X_with_labels[:, -1] == i)
            count_C = np.count_nonzero(coreset[:, -1] == i)
            coreset[coreset[:, -1] == i, -1] = count_X / count_C
        # ? 这里有疑问
        # print(coreset.shape)
        # print(len(indices))
        # print(indices)
        # print(weights.shape)
        #print(weights[indices].shape)
        if weights is not None:
            weights = np.array(weights)
            coreset[:,-1] = coreset[:,-1]*weights[indices]
        return coreset, coreset_y[:,:-1].reshape(-1,1)
    except Exception as e:
        print(e)
        print(indices)
        traceback.print_exc()
        return None




# %%
def coreset_importance(X,y=None,weights = None,problem_attributes=None, coreset_size=1000,initial_centers=None,loss_function=None,is_kmeans=False):
    # todo implement importance sample
    
    try:
        simple_version = 1
        if simple_version:
            
            if is_kmeans:
                if weights is None:
                    weights = np.ones((X.shape[0],1)).flatten()
                coreset_gen = coresets.KMeansCoreset(X, w=np.array(weights),init = initial_centers)
                C,w=coreset_gen.generate_coreset(coreset_size)
                #print("???")
                coreset = np.hstack((C, w.reshape(-1,1)))
                
                # ? 这里的y不对
                return coreset, np.full((coreset.shape[0],1),-1)
            else:
                # QR-decompositon of D_w X
                if weights is None:
                    weights = np.ones(X.shape[0])
                weights = np.array(weights)
                # D_w = np.diag(weights)

                # print(1)
                # X_w = np.dot(D_w,X)
                X_w = weights[:, np.newaxis] * X

                #print(2)
                Q,R = np.linalg.qr(X_w)
               
                # using ||Q_i||^2 +w_i/\sum w_i as sensitivity
                # select Q_i
                column_norms = np.linalg.norm(Q, axis=1)
                #print(4)
                sensitivity = column_norms + weights/np.sum(weights)
                prob = sensitivity/np.sum(sensitivity)
                #print(5)
                sample_index = np.random.choice(X.shape[0],p=prob,size=coreset_size)
                C = X[sample_index]

                weight_vec = 1/(prob+1e-10)/coreset_size
                w = weight_vec[sample_index]*np.array(weights)[sample_index]
                coreset = np.hstack((C,w.reshape(-1,1)))
                coreset_y = y[sample_index]
                return coreset, coreset_y.reshape(-1,1)

        else:
            losses = loss_function(X,y,problem_attributes=problem_attributes,theta=initial_centers)
            prob = losses/np.sum(losses)
            print(len(prob))
            print(losses)
            indices = np.arange(len(losses))
            
            sample_index = np.random.choice(indices,p=prob,size=coreset_size)
            
            C = X[sample_index]
            tol = 1e-10
            print(2)
            weight_vec = 1/(prob+1e-10)/coreset_size
            print(len(weight_vec))
            w = weight_vec[sample_index]
            if weights is not None:
                w = w* weights[sample_index]
            print(len(w))
            coreset = np.hstack((C,w.reshape(-1,1)))
            coreset_y = y[sample_index]
            return coreset, coreset_y.reshape(-1,1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None
# %%
def coreset_construction(X, initial_centers, problem_attributes=None, y=None, outliers_ratio=0.1, 
                         method='uniform', coreset_size=1000, all_outliers=False,loss_function = kmeans_loss,is_kmeans=False):
    X_copy = X.copy()
    y_copy = y.copy()
    # print(X_copy.shape)
    # print(y_copy.shape)
    n_samples, n_features = X.shape
    #print(method)
    # 计算每个样本属于每个组件的概率
    # 根据outliers_ratio分割样本
    # print(method)
    if method == "all_ring":
        C = coreset_ring(X_copy,y_copy,problem_attributes=problem_attributes,coreset_size=coreset_size,initial_centers=initial_centers,loss_function=loss_function)
        C_x, C_y = C
        return C_x, C_y.flatten()
    if method == "all_uniform":
        C = coreset_uniform(X_copy, y=y_copy,coreset_size=coreset_size)
        C_x, C_y = C
        return C_x, C_y.flatten()
    if method == "all_GSP":
        C = coreset_GSP(X_copy, y=y_copy, problem_attributes = problem_attributes, 
                        coreset_size=coreset_size, initial_centers=initial_centers,loss_function=loss_function)
        C_x, C_y = C
        return C_x, C_y.flatten()
       # return C
    if method == "all_importance":
        #print("?")
        C = coreset_importance(X_copy, y=y_copy, problem_attributes = problem_attributes, coreset_size=coreset_size, initial_centers=initial_centers,loss_function=loss_function
                               ,is_kmeans=is_kmeans)
        C_x, C_y = C
        #print(len(C_x))
        
        return C_x, C_y.flatten()
        #return C
    #print(0)
    losses = loss_function(X_copy, y=y_copy,theta = initial_centers, problem_attributes=initial_centers.shape[0])
    #print(1)
    index = losses.argsort()
    X_si = X_copy[index[:int(n_samples * (1 - outliers_ratio))]]
    X_so = X_copy[index[int(n_samples * (1 - outliers_ratio)):]]
    Y_si = y_copy[index[:int(n_samples * (1 - outliers_ratio))]]
    Y_so = y_copy[index[int(n_samples * (1 - outliers_ratio)):]]
    
    # ? here change
    n_C_so = int(coreset_size * outliers_ratio*2.5)
    n_C_si = coreset_size - n_C_so

    
    
    if n_C_so > len(X_so):
        n_C_so = len(X_so)
        n_C_si = coreset_size - n_C_so
    #print(2)
    #print(n_C_si,n_C_so)
    if all_outliers == True:
        C_so = coreset_uniform(X_so,y=Y_so,coreset_size=len(X_so))
    C_so = coreset_uniform(X_so, y=Y_so,coreset_size=n_C_so)
    #print(3)
    if method == 'uniform':
        C_si = coreset_uniform(X_si, y=Y_si,coreset_size=n_C_si)    
        #print("??")   
    elif method == 'GSP':
        C_si = coreset_GSP(X_si, y=Y_si, coreset_size=n_C_si, 
                           problem_attributes=problem_attributes, 
                           initial_centers=initial_centers,loss_function=loss_function)
        #print(C_si)
    elif method == "importance":
        C_si = coreset_importance(X_si, y=Y_si,coreset_size=n_C_si, problem_attributes=problem_attributes, initial_centers=initial_centers,loss_function=loss_function,
                                  is_kmeans=is_kmeans)
    elif method == "ring":
        C_si = coreset_ring(X_si, y=Y_si, coreset_size=n_C_si, problem_attributes=problem_attributes, initial_centers=initial_centers,loss_function=loss_function)
    #print(5)
    C_si_x,C_si_y = C_si
    C_so_x,C_so_y = C_so
    #print(6)

    C_x = np.vstack((C_si_x,C_so_x))
    
    #rint(7)
    C_y = np.vstack((C_si_y,C_so_y))

    return C_x,C_y.flatten()

class BinaryTree__:
    def __init__(self,bucketsize,theta_tuta,method="uniform",is_kmeans=True,problem_attributes=None,loss_function=kmeans_loss):
        self.root = None
        self.bucketsize = bucketsize
        self.leaves = []
        self.is_kmeans = is_kmeans
        self.method = method
        self.hotbucket = None
        self.theta_tuta = theta_tuta
        self.problem_attributes = problem_attributes
        self.loss_function = loss_function
        self.lock = True
    def freeze(self):
        self.lock = True
    def unfreeze(self):
        self.lock = False
    def fully_update(self):
        #print("fully update")
        depths = [leaf.depth for leaf in self.leaves]
        depths_np = np.array(depths)
        max_i = np.where(depths_np == np.max(depths_np))[0]
        min_i = np.where(depths_np == np.min(depths_np))[0]

        if np.max(depths_np)> np.min(depths_np):
            # print(max_i)
            # print(min_i)
            to_be_updated = [self.leaves[i] for i in max_i]
            parents = [self.leaves[i] for i in min_i]
        else:
            to_be_updated = self.leaves
            parents = []
        updated = []
        while len(to_be_updated) > 1:
            #print("root:")
            # print(len(self.root.data_instances))
            # print("To be updated")
            # for leaf in to_be_updated:
                #print(len(leaf.data_instances))
            for leaf in to_be_updated:
                if (leaf in updated) | (leaf.sibling in updated):
                    continue
                #print("start update")
                update_bucket(self,leaf,once=True)
                updated.append(leaf)
                parents.append(leaf.parent)
            # print("parents:")
            # for parent in parents:
            #     print(len(parent.data_instances))
            to_be_updated = parents
            parents = []
            updated = []
        return
        
       
    def insert_bucket(self,init=False):
        new_bucket = bucket(self.bucketsize)
        if self.root is None:
            new_bucket.depth = 0
            self.root = new_bucket
        else:
            if (len(self.leaves) & (len(self.leaves)-1)==0):
                leaf = self.leaves[0]
            else:
                #find in second last level

                edge = self.leaves[-1]
                while(edge == edge.parent.right_child):
                    #print("?")
                    edge = edge.parent
                edge = edge.parent.right_child
                while(edge.left_child is not None):
                    edge = edge.left_child
                leaf = edge
            
            #print(type(old_parent))
            new_parent = bucket(self.bucketsize)
            new_parent.depth = leaf.depth
            new_bucket.depth = leaf.depth + 1
            leaf.depth = leaf.depth + 1
            
            
            old_parent = leaf.parent
            new_parent.left_child = leaf
            leaf.parent = new_parent
            new_parent.right_child = new_bucket
            new_bucket.parent = new_parent

            leaf.sibling = new_bucket
            new_bucket.sibling = leaf

            if old_parent is not None:
                new_parent.parent = old_parent
                if old_parent.right_child == leaf:
                    old_parent.right_child = new_parent
                else:
                    old_parent.left_child = new_parent
                
                old_parent.left_child.sibling = old_parent.right_child
                old_parent.right_child.sibling = old_parent.left_child
            else:
                self.root = new_parent
        self.leaves.append(new_bucket)
        self.hotbucket = new_bucket
        if init == False:
            new_bucket.data_instances=np.array([[]])
            new_bucket.data_instances_y=np.array([[]])
        # print(type(self.hotbucket.data_instances))
    def delete(self,leaf_bucket,table):

        rightmost_leaf = self.leaves[-1]
        #hotbucket = self.hotbucket
        #print((self.leaves.index(rightmost_leaf)))
        #print(rightmost_leaf == self.leaves[-1])
        #print("check 0")
        #check(table,self)
        if rightmost_leaf != leaf_bucket:
            #print(f"add all data in bucket {self.leaves.index(hotbucket)} data to {self.leaves.index(leaf_bucket)}")
            leaf_bucket.data_instances = rightmost_leaf.data_instances
            rightmost_leaf.data_instances = np.array([[]])

            leaf_bucket.data_instances_y = rightmost_leaf.data_instances_y
            rightmost_leaf.data_instances_y = np.array([[]])
            #print(f"bucket {self.leaves.index(leaf_bucket)} has {len(leaf_bucket.data_instances)} point after add")
            #print(f"bucket {self.leaves.index(hotbucket)} has {len(hotbucket.data_instances)} point after add")
            
            #print(f"73 is refering to {self.leaves.index(table.array[73][1])} before")
            for i,item in enumerate(table.array):
                if item[1] == rightmost_leaf:
                    #print(item)
                    table.array[i] = (item[0],leaf_bucket,item[2],item[3])

        grand = rightmost_leaf.parent.parent
        if grand is None:
            self.root = rightmost_leaf.parent.left_child
        else:
            if grand.right_child == rightmost_leaf.parent:
                grand.right_child = rightmost_leaf.parent.left_child
            else:
                grand.left_child = rightmost_leaf.parent.left_child
            grand.left_child.sibling = grand.right_child
            grand.right_child.sibling = grand.left_child
        self.leaves.pop(self.leaves.index(rightmost_leaf))

        self.hotbucket = None
    def trans(self):
        for leaf in self.leaves:
            leaf.data_instances = np.array(leaf.data_instances).reshape(len(leaf.data_instances),-1)
            leaf.data_instances_y = np.array(leaf.data_instances_y).reshape(len(leaf.data_instances_y),1)
        return
    def summary(self):
        i = 0
        for leaf in self.leaves:
            i +=1
            print(f"leaf {i} has {len(leaf.data_instances)}")
class bucket:
    def __init__(self, bucketsize):
        # self.data_instances = np.array([[]])
        # self.data_instances_y = np.array([[]])
        self.data_instances = []
        self.data_instances_y = []
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.parent = None
        self.bucketsize = bucketsize
        self.depth = -1

    def is_full(self):
        return len(self.data_instances) >= self.bucketsize
    def is_half_full(self):
        return len(self.data_instances) >= self.bucketsize/2
    def insert(self, instance, instance_y,init = False):
        # if len(instance.shape) == 1:
        #     instance = np.expand_dims(instance, axis=0)
       
        if init:
            if len(self.data_instances) < self.bucketsize:
                    # print(self.data_instances.shape,instance.shape)
                if type(instance[0]) == list:
                    self.data_instances.extend(instance)
                    self.data_instances_y.extend(instance_y)
                else:
                    self.data_instances.append(instance)
                    self.data_instances_y.append(instance_y)
                
            else:
                print("Bucket is full. Cannot insert more data instances.")
        else:
            # print("?????????????")
            if self.data_instances.size == 0:
                self.data_instances = np.array(instance)
                
                self.data_instances_y = np.array(instance_y)
            elif len(self.data_instances) < self.bucketsize:
                    # print(self.data_instances.shape,instance.shape)
                self.data_instances = np.vstack((self.data_instances, instance))
                #print(self.data_instances_y.shape,instance_y.shape)
                self.data_instances_y = np.vstack((self.data_instances_y,np.array(instance_y)))
            else:
                print("Bucket is full. Cannot insert more data instances.")
        #print(type(self.data_instances))
    def delete(self,x):
        #? only call on the leaves ,so no weight
        # update 现在最后一层也有weight
        # if len(x.shape) == 1:
        #     x = np.expand_dims(x, axis=0)
        # x = x.flatten()
        
        for i in range(len(self.data_instances)):
            
            if (self.data_instances[i][:-1] == x[:,:-1]).all():
                #print("delete ",i)
                self.data_instances = np.delete(self.data_instances, i, axis=0)
                self.data_instances_y = np.delete(self.data_instances_y, i, axis=0)
                return
           
        # if len(x.shape) == 1:
        #     x = np.expand_dims(x, axis=0)
        # indices = np.where((self.data_instances == x).all(axis=1))[0]
        # indices = np.argmax(indices)
        # print("in delete bucket :",indices)
        # self.data_instances = np.delete(self.data_instances, indices, axis=0)
    #? 这儿的数据结构好像也没必要，也就多了常数因子的储存，不过时间应该也是常数因子
    # 没使用


class table:
    def __init__(self, theta_tuta, z = None):
        self.array = []
        self.critical_pointer = None
        self.z = z
        #self.z_prop = z_prop
        self.theta_tuta = theta_tuta
        #self.bucketsize = bucketsize

            


# %%
def update_bucket(tree, bucket,once=False):
    #update the tree using the bucket_idx from bottom to top
    #print(bucket_idx)
    #print(1)

    if tree.lock:
        #print("Locked")
        return
    loss_function = tree.loss_function
    method = tree.method
    is_kmeans = tree.is_kmeans
    #new
    #print(2)
    if bucket.parent is None:
        return
    # if bucket.parent.left_child == bucket:
    #     sibling = bucket.parent.right_child
    # else:
    #     sibling = bucket.parent.left_child
    sibling = bucket.sibling
    # if (len(bucket.data_instances) + len(sibling.data_instances)) < bucket.bucketsize:
    #     #print("not enough instance to update!")
    #     return
    #print(3)
   
    # the bottom buckets' weight are all 1
    # deprecated
    #level = -1
    while bucket.parent is not None:
        # level += 1
        
        parent = bucket.parent
        sibling = bucket.sibling
        #print(f"at level {level}")
        # print(type(sibling.data_instances))
        # print(type(bucket.data_instances))
        
        # weights_ = [item[-1] for item in bucket.data_instances]+[item[-1] for item in sibling.data_instances]
        # data_ = [item[:-1] for item in bucket.data_instances]+[item[:-1] for item in sibling.data_instances]
        # try:
        #     data_y = bucket.data_instances_y + sibling.data_instances_y
        # except:
        #     print(bucket.data_instances_y)
        if (bucket.data_instances.shape[1] == 0)&(sibling.data_instances.shape[1] == 0):
            parent.data_instances = np.array([[]])
            parent.data_instances_y = np.array([[]])
            bucket = parent
            if once:
                break
            continue
        if sibling.data_instances.shape[1] == 0:
            weights_ = bucket.data_instances[:,-1]
            data_ = bucket.data_instances[:,:-1]
            data_y = bucket.data_instances_y
        elif bucket.data_instances.shape[1] == 0:
            weights_ = sibling.data_instances[:,-1]
            data_ = sibling.data_instances[:,:-1]
            data_y = sibling.data_instances_y
        else:
            #print(bucket.data_instances.shape,sibling.data_instances.shape)
            try:
                weights_ = np.hstack((bucket.data_instances[:,-1],sibling.data_instances[:,-1]))
                data_ = np.vstack((bucket.data_instances[:,:-1],sibling.data_instances[:,:-1]))
                data_y = np.vstack((bucket.data_instances_y,sibling.data_instances_y))
            except:
                print(bucket.data_instances.shape,sibling.data_instances.shape)
                print(bucket.data_instances)
        #print(len(weights_))
        if len(weights_) >  bucket.bucketsize:
            size = int(bucket.bucketsize)
            if method == "uniform":
                # update the parent bucket using uniform sampling method,using children buckets' data and weights
               
                parent.data_instances,parent.data_instances_y = coreset_uniform(X=data_,y=data_y, weights=weights_,coreset_size=size)     
            elif method == "GSP":
                # update the parent bucket using GSP sampling method,using children buckets' data and weights
                parent.data_instances,parent.data_instances_y = coreset_GSP(data_, y=data_y,weights=weights_,
                                                                            problem_attributes=tree.problem_attributes,
                                                                            coreset_size=size,initial_centers=tree.theta_tuta,
                                                                            loss_function=loss_function)
            elif method == "importance":
                parent.data_instances,parent.data_instances_y = coreset_importance(data_, y=data_y,weights=weights_,
                                                                            problem_attributes=tree.problem_attributes,
                                                                            coreset_size=size,initial_centers=tree.theta_tuta,
                                                                            loss_function=loss_function,is_kmeans=is_kmeans)
            elif method == "ring":
                parent.data_instances,parent.data_instances_y = coreset_ring(data_, y=data_y,weights=weights_,
                                                                            problem_attributes=tree.problem_attributes,
                                                                            coreset_size=size,initial_centers=tree.theta_tuta,
                                                                            loss_function=loss_function)
                
            else:
                print("Invalid method")

        else:
            
            parent.data_instances = np.hstack((data_,weights_.reshape(-1,1)))
           
            parent.data_instances_y = data_y
        if test:
            print(parent.data_instances.shape,parent.data_instances_y.shape)
        
        bucket = parent
        if once:
            break
    #print("Bucket {} is updated".format(bucket_idx))
    
    return 
    pass

# old version
# def update_bucket(tree, bucket,once=False):
#     #update the tree using the bucket_idx from bottom to top
#     #print(bucket_idx)
#     #print(1)

#     if tree.lock:
#         #print("Locked")
#         return
#     loss_function = tree.loss_function
#     method = tree.method
#     is_kmeans = tree.is_kmeans
#     #new
#     #print(2)
#     if bucket.parent is None:
#         return
#     if bucket.parent.left_child == bucket:
#         sibling = bucket.parent.right_child
#     else:
#         sibling = bucket.parent.left_child
#     # sibling = bucket.sibling
#     # if (len(bucket.data_instances) + len(sibling.data_instances)) < bucket.bucketsize:
#     #     #print("not enough instance to update!")
#     #     return
#     #print(3)
   
#     # the bottom buckets' weight are all 1
#     # deprecated
#     #level = -1
#     while bucket.parent is not None:
#         #level += 1
#         parent = bucket.parent
#         if sibling.data_instances.shape[1] == 0:
#             weights_ = bucket.data_instances[:,-1]
#             data_ = bucket.data_instances[:,:-1]
#             data_y = bucket.data_instances_y
#         else:
#             #print(bucket.data_instances.shape,sibling.data_instances.shape)
#             weights_ = np.hstack((bucket.data_instances[:,-1],sibling.data_instances[:,-1]))
#             data_ = np.vstack((bucket.data_instances[:,:-1],sibling.data_instances[:,:-1]))
#             data_y = np.vstack((bucket.data_instances_y,sibling.data_instances_y))
#         #print(len(weights_))
#         if len(weights_) >=  bucket.bucketsize:
#             size = int(bucket.bucketsize)
#             if method == "uniform":
#                 # update the parent bucket using uniform sampling method,using children buckets' data and weights
#                 parent.data_instances,parent.data_instances_y = coreset_uniform(X=data_,y=data_y, weights=weights_,coreset_size=size)     
#             elif method == "GSP":
#                 # update the parent bucket using GSP sampling method,using children buckets' data and weights
#                 parent.data_instances,parent.data_instances_y = coreset_GSP(data_, y=data_y,weights=weights_,
#                                                                             problem_attributes=tree.problem_attributes,
#                                                                             coreset_size=size,initial_centers=tree.theta_tuta,
#                                                                             loss_function=loss_function)
#             elif method == "importance":
#                 parent.data_instances,parent.data_instances_y = coreset_importance(data_, y=data_y,weights=weights_,
#                                                                             problem_attributes=tree.problem_attributes,
#                                                                             coreset_size=size,initial_centers=tree.theta_tuta,
#                                                                             loss_function=loss_function,is_kmeans=is_kmeans)
#             else:
#                 print("Invalid method")
#         else:
#             parent.data_instances = np.hstack((data_,weights_.reshape(-1,1)))
#             parent.data_instances_y = data_y
        
        
#         bucket = parent
#         if once:
#             break
#     #print("Bucket {} is updated".format(bucket_idx))
#     return tree
#     pass

# %%
#new version
def update_z(delta,table,tree):
    theta_tuta = table.theta_tuta
    if delta == 0 :
        return
    if delta > 0:
        n_delta = ceil(delta)
        if len(table.array) <= table.z:
            table.z = table.z+ n_delta
            return
        else:
            for i in range(n_delta):
                #print("CHECK1 in update_z")
                #check(table,tree)

                table_index = table.critical_pointer+i
                target_value = table.array[table_index][0]
                target_bucket = table.array[table_index][1]
                #target_bucket = tree.leaves[target_index]
                retrieved_X = table.array[table_index][2]
                retrieved_y = table.array[table_index][3]
                
                table.array[table_index]=(target_value,-1,retrieved_X,retrieved_y)
                #? 没考虑 delete X,y
                # print("critical pointer",table.critical_pointer)
                # print("target_bucket",target_bucket)
                target_bucket.delete(retrieved_X)

                update_list = [target_bucket]
                if target_bucket.is_half_full() == 0:
                    if tree.hotbucket is None:
                        tree.hotbucket = target_bucket
                    elif tree.hotbucket != target_bucket:
                        if tree.leaves.index(tree.hotbucket)!= len(tree.leaves)-1:
                            update_list.append(tree.hotbucket)
                        #print("CHECK1.5 in update_z")
                        #check(table,tree)
                        
                        tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
                        target_bucket.data_instances = np.array([[]])
                        target_bucket.data_instances_y =np.array([[]])
                        for j,item in enumerate(table.array):
                            if item[1] == target_bucket:
                                table.array[j] =(item[0],tree.hotbucket,item[2],item[3])
                        tree.delete(target_bucket,table)

                        rightmost_leaf = tree.leaves[-1]
                        grand = rightmost_leaf.parent.parent
                        if grand is None:
                            update_list = [tree.root.right_child]
                        else:
                            if grand.right_child == rightmost_leaf.parent:
                                update_list.append(grand.left_child)
                            else:
                                update_list.append(grand.right_child)
                update_list = set(update_list)
                for item in update_list:
                    if item in tree.leaves:
                    
                        update_bucket(tree,item)
                #print("CHECK2 in update_z")
                
                #update_bucket(tree,target_bucket)
                #table.array[table_index]=(target_value,-1,retrieved_X)
                #check(table,tree)
            table.z = table.z + n_delta
            table.critical_pointer = table.critical_pointer + n_delta
    else:
        ## delta < 0 
        n_delta = floor(delta)
        if n_delta + table.z <= 0:
            print("invalid delta")
            return
        if len(table.array) <= table.z:
            table.z = table.z + n_delta
            return
        else:
            #print(table.z)
            #print(len(table.array))
            for i in range(-n_delta):
                hotbucket = tree.hotbucket
                if hotbucket is None:
                    tree.insert_bucket()
                hotbucket = tree.hotbucket


                table_index = table.critical_pointer - i-1
                target_value = table.array[table_index][0]
                insert_x = table.array[table_index][2]
                insert_y = table.array[table_index][3]
                #print(insert_y.shape)
                hotbucket.insert(insert_x,insert_y)
                table.array[table_index] = (target_value, hotbucket,insert_x,insert_y)
                
                update_bucket(tree,hotbucket)
                if hotbucket.is_half_full():
                    tree.hotbucket = None
               
                #tree.choose_hot_bucket()
            table.z = table.z  - n_delta
            table.critical_pointer = table.critical_pointer - n_delta
    return 

       
# old version
# def update_z(delta,table,tree):
#     theta_tuta = table.theta_tuta
#     if delta == 0 :
#         return
#     if delta > 0:
#         n_delta = ceil(delta)
#         if len(table.array) <= table.z:
#             table.z = table.z+ n_delta
#             return
#         else:
#             for i in range(n_delta):
#                 #print("CHECK1 in update_z")
#                 #check(table,tree)

#                 table_index = table.critical_pointer-1-i
#                 target_value = table.array[table_index][0]
#                 target_bucket = table.array[table_index][1]
#                 #target_bucket = tree.leaves[target_index]
#                 retrieved_X = table.array[table_index][2]
#                 retrieved_y = table.array[table_index][3]
                
#                 table.array[table_index]=(target_value,-1,retrieved_X,retrieved_y)
#                 #? 没考虑 delete X,y
#                 target_bucket.delete(retrieved_X)

#                 update_list = [target_bucket]
#                 if target_bucket.is_half_full() == 0:
#                     if tree.hotbucket is None:
#                         tree.hotbucket = target_bucket
#                     elif tree.hotbucket != target_bucket:
#                         if tree.leaves.index(tree.hotbucket)!= len(tree.leaves)-1:
#                             update_list.append(tree.hotbucket)
#                         #print("CHECK1.5 in update_z")
#                         #check(table,tree)
                        
#                         tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
#                         target_bucket.data_instances = np.array([[]])
#                         target_bucket.data_instances_y = np.array([[]])
#                         #print("CHECK1.6 in update_z")
#                         #check(table,tree)
#                         #print(f"hotbucket {tree.leaves.index(tree.hotbucket)} has {len(tree.hotbucket.data_instances)} point after")
#                         #print(f"target_bucket {tree.leaves.index(target_bucket)} has {len(target_bucket.data_instances)} point after")


#                         #print(f"73 is refering to {tree.leaves.index(table.array[73][1])}")
#                         #print(f"79 is refering to {tree.leaves.index(table.array[79][1])}")
#                         for i,item in enumerate(table.array):
#                             if item[1] == target_bucket:
#                                 table.array[i] =(item[0],tree.hotbucket,item[2],item[3])
#                          #       print(f"{i} refer from {tree.leaves.index(target_bucket)} to {tree.leaves.index(tree.hotbucket)}")
#                         #print(f"73 is refering to {tree.leaves.index(table.array[73][1])}")

#                         #print("CHECK1.7 in update_z")
#                         #check(table,tree)
#                         #print(f"79 is refering to {tree.leaves.index(table.array[79][1])}")
#                         tree.delete(target_bucket,table)

#                         rightmost_leaf = tree.leaves[-1]
#                         grand = rightmost_leaf.parent.parent
#                         if grand is None:
#                             update_list = [tree.root.right_child]
#                         else:
#                             if grand.right_child == rightmost_leaf.parent:
#                                 update_list.append(grand.left_child)
#                             else:
#                                 update_list.append(grand.right_child)
#                 update_list = set(update_list)
#                 for item in update_list:
#                     if item in tree.leaves:
#                         update_bucket(tree,item)
#                 #print("CHECK2 in update_z")
                
#                 #update_bucket(tree,target_bucket)
#                 #table.array[table_index]=(target_value,-1,retrieved_X)
#                 #check(table,tree)
#             table.z = table.z + n_delta
#             table.critical_pointer = table.critical_pointer - n_delta
#     else:
#         ## delta < 0 
#         n_delta = floor(delta)
#         if n_delta + table.z <= 0:
#             print("invalid delta")
#             return
#         if len(table.array) <= table.z:
#             table.z = table.z + n_delta
#             return
#         else:
#             #print(table.z)
#             #print(len(table.array))
#             for i in range(-n_delta):
#                 hotbucket = tree.hotbucket
#                 if hotbucket is None:
#                     tree.insert_bucket()
#                 hotbucket = tree.hotbucket


#                 table_index = table.critical_pointer + i
#                 target_value = table.array[table_index][0]
#                 insert_x = table.array[table_index][2]
#                 insert_y = table.array[table_index][3]
#                 #print(insert_y.shape)
#                 hotbucket.insert(insert_x,insert_y)
#                 table.array[table_index] = (target_value, hotbucket,insert_x,insert_y)
                
#                 if hotbucket.is_half_full():
#                     tree.hotbucket = None
#                 update_bucket(tree,hotbucket)
#                 #tree.choose_hot_bucket()
#             table.z = table.z  + n_delta
#             table.critical_pointer = table.critical_pointer - n_delta
#     return 

        
        

# %%
# ? 没维护sibling
#new verison
def delete_data(X,y,table,tree):
    #print(X.shape)
    theta_tuta = table.theta_tuta
    problem_attributes = tree.problem_attributes
    loss_function = tree.loss_function
    old_loss = loss_function(X[:,:-1], y, theta = theta_tuta,problem_attributes=problem_attributes)
    #print(old_loss)
    
    #? here remains doubt, find X by loss in table ,assume no same loss, if duplicated, both refer to the tree
    # find X in the table
    i = bisect.bisect_left([item[0] for item in table.array],old_loss)
    #print(i)
    if (table.array[i][0] != old_loss):
        print("Not Found!")
        print(old_loss)
        print(table.array[i-1][0])
        print(table.array[i][0])
        print(table.array[i+1][0])

        return
    # print(i)
    # print(table.array[i-1][0])
    # print(table.array[i][0])
    # print(table.array[i+1][0])
    # print(table.critical_pointer)
    #print("Found!")

    if len(table.array) < table.z:
        table.array.pop(i)
    else:
        if i >= table.critical_pointer:
            #print(i)
            target_bucket = table.array[i][1]

            table.array.pop(i)

            target_bucket.delete(X)
            # print(type(target_bucket.data_instances))

            if target_bucket.is_half_full() == 0:
                if tree.hotbucket is None:
                    tree.hotbucket = target_bucket
                    # print(1)
                elif tree.hotbucket != target_bucket:
                    # print(2)
                    tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
                    target_bucket.data_instances = np.array([[]])
                    target_bucket.data_instances_y = np.array([[]])
                    for j ,item in enumerate(table.array):
                        if item[1] == target_bucket:
                            table.array[j] =(item[0],tree.hotbucket,item[2],item[3])
                    tree.delete(target_bucket,table)
                # print(3)
            # print(type(target_bucket.data_instances))
            update_bucket(tree, target_bucket)
            
        else:
            # print("???")
            target_bucket = table.array[table.critical_pointer][1]
            # print(type(target_bucket.data_instances))
            #target_bucket = tree.leaves[target_index]
            target_value = table.array[table.critical_pointer][0]
            #? here needs to find x by loss in the tree again? maybe store this? both not good
            retrieved_X =  table.array[table.critical_pointer][2]
            retrieved_y =  table.array[table.critical_pointer][3]
            # print(table.array[table.critical_pointer-1][1])
            # print(target_bucket)
            # print(table.array[table.critical_pointer+1][1])

            target_bucket.delete(retrieved_X)
           
                
             
            test = 0
            if target_bucket.is_half_full() == 0:
                if tree.hotbucket is None:
                    # print(11)
                    tree.hotbucket = target_bucket
                elif tree.hotbucket != target_bucket:
                    # print(22)
                    test = 1
                    tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
                    target_bucket.data_instances = np.array([[]])
                    target_bucket.data_instances_y = np.array([[]])
                    for j, item in enumerate(table.array):
                        if item[1] == target_bucket:
                            #??????????? item =  wrong
                            table.array[j] =(item[0],tree.hotbucket,item[2],item[3])
                    # print(table.array[table.critical_pointer-1][1])
                    # print(table.array[table.critical_pointer][1])
                    # print(table.array[table.critical_pointer+1][1])
                    tree.delete(target_bucket,table)
            #? like update_z need update more points
            # print(33)
            # print(type(target_bucket.data_instances))
            update_bucket(tree, target_bucket)
            
            table.array[table.critical_pointer] = (target_value,-1,retrieved_X,retrieved_y)
            # if test:
            #     print(table.array[table.critical_pointer-1][1])
            #     print(table.array[table.critical_pointer][1])
            #     print(table.array[table.critical_pointer+1][1])
            table.array.pop(i)
            #print(i)
            # if test:
            #     print(table.array[table.critical_pointer-1][1])
            #     print(table.array[table.critical_pointer][1])
            #     print(table.array[table.critical_pointer+1][1])
            #table.critical_pointer -= 1
    #print("delete done")
    return

#old version
# def delete_data(X,y,table,tree):
#     #print(X.shape)
#     theta_tuta = table.theta_tuta
#     problem_attributes = tree.problem_attributes
#     loss_function = tree.loss_function
#     old_loss = loss_function(X[:-1].reshape(1,-1), y, theta = theta_tuta,problem_attributes=problem_attributes)
#     #print(old_loss)
    
#     #? here remains doubt, find X by loss in table ,assume no same loss, if duplicated, both refer to the tree
#     # find X in the table
#     i = bisect.bisect_left([item[0] for item in table.array],old_loss)
    
#     if (table.array[i][0] != old_loss):
#         print("Not Found!")
#         return
#     # print(i)
#     # print(table.array[i-1][0])
#     # print(table.array[i][0])
#     # print(table.array[i+1][0])
#     # print(table.critical_pointer)
#     #print("Found!")

#     if len(table.array) < table.z:
#         table.array.pop(i)
#     else:
#         if i < table.critical_pointer:
#             #print(i)
#             target_bucket = table.array[i][1]
#             #print(target_bucket)
#             #target_index = 6
#             table.array.pop(i)
#             table.critical_pointer -= 1
#             #target_bucket = tree.leaves[target_index]
#             #print("check ?")
#             #check(table,tree)
#             target_bucket.delete(X)

#             if target_bucket.is_half_full() == 0:
#                 if tree.hotbucket is None:
#                     tree.hotbucket = target_bucket
#                 elif tree.hotbucket != target_bucket:
#                     tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
#                     target_bucket.data_instances = np.array([[]])
#                     target_bucket.data_instances = np.array([[]])
#                     for i ,item in enumerate(table.array):
#                             if item[1] == target_bucket:
#                                 table.array[i] =(item[0],tree.hotbucket,item[2],item[3])
#                     tree.delete(target_bucket,table)
                    
#             update_bucket(tree, target_bucket)
#         else:
#             target_bucket = table.array[table.critical_pointer-1][1]
#             #target_bucket = tree.leaves[target_index]
#             target_value = table.array[table.critical_pointer-1][0]
#             #? here needs to find x by loss in the tree again? maybe store this? both not good
#             retrieved_X =  table.array[table.critical_pointer-1][2]
#             retrieved_y =  table.array[table.critical_pointer-1][3]
#             target_bucket.delete(retrieved_X)

#             if target_bucket.is_half_full() == 0:
#                 if tree.hotbucket is None:
#                     tree.hotbucket = target_bucket
#                 elif tree.hotbucket != target_bucket:
#                     tree.hotbucket.insert(target_bucket.data_instances,target_bucket.data_instances_y)
#                     target_bucket.data_instances = np.array([[]])
#                     target_bucket.data_instances_y = np.array([[]])
#                     for i, item in enumerate(table.array):
#                             if item[1] == target_bucket:
#                                 #??????????? item =  wrong
#                                 table.array[i] =(item[0],tree.hotbucket,item[2],item[3])
#                     tree.delete(target_bucket,table)
#             #? like update_z need update more points
#             update_bucket(tree, target_bucket)
#             table.array[table.critical_pointer-1] = (target_value,-1,retrieved_X,retrieved_y)
#             table.array.pop(i)
#             table.critical_pointer -= 1
#     #print("delete done")
#     return

# new version       
def insert_data(X, y, table, tree,init=False):
    theta_tuta = table.theta_tuta
    problem_attributes = tree.problem_attributes
    loss_function = tree.loss_function
    #print("insert ",X.shape)
    #print(table.z)
    #print(X[:-1].reshape(1,-1).shape)
    if init:
        X = np.array(X).reshape(1,-1)
        y = np.array(y)
    new_loss = loss_function(X[:,:-1], y,theta = theta_tuta,problem_attributes=problem_attributes)
    #print(new_loss.shape)
    ## if the table is small, get it aside first
    #print([item[0] for item in table.array])
    #X = X.tolist()
    #y = y.flatten().tolist()
    if init == True:
        i = len(table.array)
    else:
        i = bisect.bisect([item[0] for item in table.array], new_loss)
    
    target_index = None
    if len(table.array) < table.z:
        #hotbucket_idx = tree.hotbucket
        # X stores its data value, not the index, prepare to be inserted into the bucket
        if init:
            table.array.append((new_loss, -1, X, y))
        else:
            table.array.insert(i, (new_loss, -1, X, y))
        if len(table.array) ==  table.z:
            #print("?????????????")
            table.critical_pointer = len(table.array)
            #print(table.critical_pointer)
        target_index = -1
    else:
        #i = bisect.bisect([item[0] for item in table.array], new_loss)
        #hotbucket_idx = tree.hotbucket
    
        #################new
        hotbucket = tree.hotbucket
        if hotbucket is None:
            tree.insert_bucket(init = init)
            #print("new bucket created!")
        hotbucket = tree.hotbucket
        #print(hotbucket)
        #target_index = hotbucket_idx

        if i >= table.critical_pointer:
            #hotbucket_idx = tree.hotbucket 
            if init:
                table.array.append((new_loss, hotbucket, X, y))
            else:
                table.array.insert(i, (new_loss, hotbucket, X, y))
            #table.critical_pointer += 1
            # print(X.shape)
            # print(y)
            hotbucket.insert(X,y,init = init)
            if hotbucket.is_half_full():
                tree.hotbucket = None
            update_bucket(tree, hotbucket)
            #tree.choose_hot_bucket()
        else:
            table.array.insert(i, (new_loss, -1, X, y))
            #hotbucket_idx = tree.hotbucket
            change_x = table.array[table.critical_pointer][2]
            change_y = table.array[table.critical_pointer][3]
            #print(hotbucket_idx)
            #print(hotbucket)
            # print(change_x.shape)
            # print(change_y)
            hotbucket.insert(change_x,change_y,init = init)
            originl_loss = table.array[table.critical_pointer][0]
            table.array[table.critical_pointer] = (originl_loss, hotbucket,change_x,change_y)

            if hotbucket.is_half_full():
                tree.hotbucket = None
            #? here????????
            update_bucket(tree, hotbucket)
            #tree.choose_hot_bucket()
            #table.critical_pointer += 1
    

    #print("insert done")
    return target_index

# %%
#old version
# def insert_data(X, y, table, tree,init=False):
#     theta_tuta = table.theta_tuta
#     problem_attributes = tree.problem_attributes
#     loss_function = tree.loss_function
#     #print("insert ",X.shape)
#     #print(table.z)
#     #print(X[:-1].reshape(1,-1).shape)
#     new_loss = loss_function(X[:-1].reshape(1,-1), y,theta = theta_tuta,problem_attributes=problem_attributes)
#     #print(new_loss.shape)
#     ## if the table is small, get it aside first
#     #print([item[0] for item in table.array])
#     if init == True:
#         i = 0
#     else:
#         i = bisect.bisect([item[0] for item in table.array], new_loss)
    
#     target_index = None
#     if len(table.array) < table.z:
#         #hotbucket_idx = tree.hotbucket
#         # X stores its data value, not the index, prepare to be inserted into the bucket
#         table.array.insert(i, (new_loss, -1, X, y))
#         if len(table.array) ==  table.z:
#             #print("?????????????")
#             table.critical_pointer = 0
#             #print(table.critical_pointer)
#         target_index = -1
#     else:
#         #i = bisect.bisect([item[0] for item in table.array], new_loss)
#         #hotbucket_idx = tree.hotbucket
    
#         #################new
#         hotbucket = tree.hotbucket
#         if hotbucket is None:
#             tree.insert_bucket()
#             #print("new bucket created!")
#         hotbucket = tree.hotbucket
#         #print(hotbucket)
#         #target_index = hotbucket_idx

#         if i <= table.critical_pointer:
#             #hotbucket_idx = tree.hotbucket 
#             table.array.insert(i, (new_loss, hotbucket, X, y))
#             table.critical_pointer += 1
#             # print(X.shape)
#             # print(y)
#             hotbucket.insert(X.flatten().reshape(1,-1),y)
#             if hotbucket.is_half_full():
#                 tree.hotbucket = None
#             update_bucket(tree, hotbucket)
#             #tree.choose_hot_bucket()
#         else:
#             table.array.insert(i, (new_loss, -1, X, y))
#             #hotbucket_idx = tree.hotbucket
#             change_x = table.array[table.critical_pointer][2]
#             change_y = table.array[table.critical_pointer][3]
#             #print(hotbucket_idx)
#             #print(hotbucket)
#             # print(change_x.shape)
#             # print(change_y)
#             hotbucket.insert(change_x.flatten().reshape(1,-1),change_y)
#             originl_loss = table.array[table.critical_pointer][0]
#             table.array[table.critical_pointer] = (originl_loss, hotbucket,change_x,change_y)

#             if hotbucket.is_half_full():
#                 tree.hotbucket = None
#             #? here????????
#             update_bucket(tree, hotbucket)
#             #tree.choose_hot_bucket()
#             table.critical_pointer += 1
    

#     #print("insert done")
#     return target_index


    

# %%
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

# Example usage

X_train, y_train = load_mnist('/home/sjj/Experiment/data/MNIST/raw', 'train')
X_test, y_test = load_mnist('/home/sjj/Experiment/data/MNIST/raw', 't10k')
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



# %%

#new version
def streaming_data(X_base, X_for_use, y_base, y_for_use, n_runs, 
table, tree, outliers_ratio, method="uniform",problem_attributes=None,loss_function=kmeans_loss):
    theta_tuta = table.theta_tuta
    losses = []
    for i in range(len(X_base)):
        insert_data(X_base[i], y_base[i], table, tree,init =True)
        # if (i% 10000 == 1):
        #     print(f"{i} data instances have been inserted")
         
    t_insert = 0
    t_delete = 0
    t_update = 0
    t_delete__ = 0
    t_insert__ = 0
    t_update__ = 0
    print("Initialize!")
    #return
    # print(table.array[table.z+1][1])
    # print(table.array[table.z+2][1])
    # print(table.array[table.z][1])
    # print(table.array[table.z-1][1])
    # print(table.array[table.z-2][1])

    iter = 0
    tree.unfreeze()
    # print("unlock!")
    # print(tree.lock)
    tree.trans()
    # print([type(leaf.data_instances) for leaf in tree.leaves])
    # print(len(tree.leaves))
    tree.fully_update()
    # print([type(leaf.data_instances) for leaf in tree.leaves])
    # print(len(tree.leaves))
    # profiler = cProfile.Profile()
    # profiler.enable()
    while min(t_insert,t_delete) <= 0.3:
        start_time = time.time()
        for i in range(n_runs):
            # print(f"insert {i}")
            # if i % 1000 == 0:
            #     print(i)
            insert_data(X_for_use[i].reshape(1,-1), y_for_use[i], table, tree)
            #size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
            #outliers = np.array([item[2] for item in table.array[:table.critical_pointer+1]])
            #outliers_y = np.array([item[3] for item in table.array[:table.critical_pointer+1]])
            #coreset_uniform(outliers, outliers_y, coreset_size=int(size))
        end_time = time.time()
        t_insert += end_time - start_time
        # print([type(leaf.data_instances) for leaf in tree.leaves])
        # print(len(tree.leaves))
        start_time = time.time()
        for i in range(n_runs):
            # print(f"delete {i}")
            # if i % 1000 == 0:
            #     print(i)
            # print(i)
            # print(table.array[table.critical_pointer-1][1])
            # print(table.array[table.critical_pointer][1])
            # print(table.array[table.critical_pointer+1][1])

            delete_data(X_for_use[i].reshape(1,-1), y_for_use[i], table, tree)
            
            
            #print()
            #size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
            #outliers = np.array([item[2] for item in table.array[:table.critical_pointer+1]])
            #outliers_y = np.array([item[3] for item in table.array[:table.critical_pointer+1]])
            #coreset_uniform(outliers, outliers_y,coreset_size=int(size))
        end_time = time.time()
        t_delete += end_time - start_time
        iter += 1
    t_insert = t_insert / iter
    t_delete = t_delete / iter
    print("insert & delete done!")
    
    
    iter = 0
    while t_update <= 0.5:
        start_time = time.time()
        for i in range(5):
            update_z(5, table, tree)
            #size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
            # outliers = np.array([item[2] for item in table.array[:table.critical_pointer+1]])
            # outliers_y = np.array([item[3] for item in table.array[:table.critical_pointer+1]])
            # print(outliers.shape)
            # coreset_uniform(outliers, outliers_y,coreset_size=int(size))
            update_z(-5, table, tree)
            # ? 这里改回来要加reshape outliers是3维 
            # size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
            # outliers = np.array([item[2] for item in table.array[:table.critical_pointer+1]])
            # outliers_y = np.array([item[3] for item in table.array[:table.critical_pointer+1]])
            # print(outliers.shape)
            # coreset_uniform(outliers, outliers_y,coreset_size=int(size))
        end_time = time.time()
        t_update += end_time - start_time
        iter += 1
    t_update = t_update / iter / 10
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()
    top_num = len(tree.root.data_instances)
    return t_insert/n_runs, t_delete/n_runs, t_update/5, top_num

#old version
# def streaming_data(X_base, X_for_use, y_base, y_for_use, n_runs, 
# table, tree, outliers_ratio, method="uniform",problem_attributes=None,loss_function=kmeans_loss):
#     theta_tuta = table.theta_tuta
#     losses = []
#     for i in range(len(X_base)):
#         insert_data(X_base[i].flatten(), y_base[i].reshape(1,-1), table, tree,init =True)
#         if (i% 10000 == 1):
#             print(f"{i} data instances have been inserted")
         
#     t_insert = 0
#     t_delete = 0
#     t_update = 0
#     t_delete__ = 0
#     t_insert__ = 0
#     t_update__ = 0
#     print("Initialize!")
#     iter = 0
#     tree.unfreeze()
#     print("unlock!")
#     print(tree.lock)
#     tree.fully_update()
#     print(len(tree.root.data_instances))
#     while min(t_insert,t_delete) <= 0.3:
#         start_time = time.time()
#         for i in range(n_runs):
#             insert_data(X_for_use[i].flatten(), y_for_use[i].reshape(1,-1), table, tree)
#             size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
#             outliers = np.array([item[2] for item in table.array[table.critical_pointer:]])
#             outliers_y = np.array([item[3] for item in table.array[table.critical_pointer:]])
#             coreset_uniform(outliers, outliers_y, coreset_size=int(size))
#         end_time = time.time()
#         t_insert += end_time - start_time

#         start_time = time.time()
#         for i in range(n_runs):
#             delete_data(X_for_use[i].flatten(), y_for_use[i], table, tree)
#             size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
#             outliers = np.array([item[2] for item in table.array[table.critical_pointer:]])
#             outliers_y = np.array([item[3] for item in table.array[table.critical_pointer:]])
#             coreset_uniform(outliers, outliers_y,coreset_size=int(size))
#         end_time = time.time()
#         t_delete += end_time - start_time
#         iter += 1
#     t_insert = t_insert / iter
#     t_delete = t_delete / iter
    
    
    
#     iter = 0
#     while t_update <= 0.5:
#         start_time = time.time()
#         for i in range(n_runs):
#             update_z(1, table, tree)
#             size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
#             outliers = np.array([item[2] for item in table.array[table.critical_pointer:]])
#             outliers_y = np.array([item[3] for item in table.array[table.critical_pointer:]])
#             coreset_uniform(outliers, outliers_y,coreset_size=int(size))
#             update_z(-1, table, tree)
#             size = (outliers_ratio)/(1-outliers_ratio)*len(tree.root.data_instances)
#             outliers = np.array([item[2] for item in table.array[table.critical_pointer:]])
#             outliers_y = np.array([item[3] for item in table.array[table.critical_pointer:]])
#             coreset_uniform(outliers, outliers_y,coreset_size=int(size))
#         end_time = time.time()
#         t_update += end_time - start_time
#         iter += 1
#     t_update = t_update / iter
    

#     top_num = len(tree.root.data_instances)
#     return t_insert/n_runs, t_delete/n_runs, t_update/n_runs, top_num


# %%kmeans_los
# run kmeans++ X_train to get theta_tuta
initial_model = KMeans(n_clusters=10, init='k-means++', n_init=1, max_iter=1, random_state=0)
X_train_test = X_train[:200]
y_train_test = y_train[:200].tolist()
X_train_weighted = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_train_test = np.hstack((X_train_test, np.ones((X_train_test.shape[0], 1)))).tolist()
initial_model.fit(X_train)
y_train_test = [[x] for x in y_train_test]
#print(X_train_test.shape)
#initial_model.fit(X_train_test)
theta_tuta = initial_model.cluster_centers_
#loss_1 =loss(X_train_test[1],theta_tuta)
#loss_0 =loss(X_train_test[0],theta_tuta)
T = table(theta_tuta, z = 10)
#oot_bucket = bucket(bucketsize=60)
binary_tree = BinaryTree__(bucketsize=60,theta_tuta=theta_tuta)
#streaming_data(X_train, T, binary_tree, theta_tuta)

for _ in range(200):
    #print(X_train_test[_])
    insert_data(X_train_test[_],y_train_test[_], T, binary_tree,init=True)




# %%
def evaluate(args):
    try:
        

        dir_name,dataset,n_level,outliers_ratio,n_runs,n_runs_original,problem_attributes, method, loss_function,seed = args
        np.random.seed(seed)
        if os.path.exists(dir_name) == False:
            os.makedirs(dir_name)
        data_,data_y,_ ,_= load_data(dataset)
        data = np.hstack((data_,np.ones((data_.shape[0],1))))
        is_kmeans = False
        if loss_function is kmeans_loss:
            is_kmeans  =True
            problem = "k_means"
            #initial model
            initial_model = KMeans(n_clusters=problem_attributes, init='k-means++', n_init=1, max_iter=1, random_state=0)
            initial_model.fit(data[:,:-1])
            #print("1")
            theta_tuta = initial_model.cluster_centers_
        elif loss_function is huber_regression_loss:
            problem = "huber_regression"
            #? here 这里没做problem_attributes的传递
            initial_model = HuberRegressor(max_iter = 3,epsilon = 1.35)
            initial_model.fit(data[:,:-1],data_y)
            theta_tuta = np.array([initial_model.scale_, initial_model.intercept_])
            theta_tuta = np.append(theta_tuta, initial_model.coef_)
        elif loss_function is logistic_loss:
            problem = "logistic"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                initial_model = LogisticRegression(max_iter = 3)
                initial_model.fit(data[:,:-1], data_y)
                theta_tuta = np.array(initial_model.intercept_)
                theta_tuta = np.append(theta_tuta, initial_model.coef_)
            # print(theta_tuta.shape)
            # print(data.shape)
            # print(initial_model.coef_.shape)
            # print(initial_model.intercept_.shape)
        
        losses = loss_function(data[:,:-1],data_y,problem_attributes = problem_attributes,
                               theta = theta_tuta)
        index = losses.argsort()[::-1]
        data_sorted = data[index]
        data_y_sorted = data_y[index]

        n_samples = data.shape[0]
        z = int(n_samples * outliers_ratio)
        bucketsize = int(n_samples/(2**(n_level-1)))+5
        print(bucketsize)
        ## initial model
        # initial_model = KMeans(n_clusters=10, init='k-means++', n_init=1, max_iter=1, random_state=0)
        # initial_model.fit(data[:,:-1])
        #print("1")
        # theta_tuta = initial_model.cluster_centers_
        #print("2")
        T = table(theta_tuta, z = z)
        #root_bucket = bucket(bucketsize=bucketsize)


        
        binary_tree = BinaryTree__(bucketsize=bucketsize,theta_tuta=theta_tuta,
                                method=method,problem_attributes=problem_attributes,is_kmeans=is_kmeans,loss_function=loss_function)
        indices= np.random.choice(data.shape[0], n_runs, replace=False)
        data_for_use = data_sorted[indices]
        y_for_use = data_y_sorted[indices]

        data_base = np.delete(data_sorted, indices, axis=0).tolist()
        
        
        y_base = np.delete(data_y_sorted,indices,axis=0).tolist()
        y_base = [[x] for x in y_base]
        
        #print(3)
        binary_tree.freeze()
        ti,td,tu,top_num= streaming_data(data_base, data_for_use, 
                                        y_base, y_for_use, n_runs, T, 
                                        binary_tree, outliers_ratio = outliers_ratio,
                                        problem_attributes=problem_attributes,
                                        loss_function=loss_function,method=method)
        original_time = 0
        print("streaming done")
        # sys.stdout.flush()
        print(top_num)
        for _ in range(n_runs_original):
            start_time = time.time()
            coreset_construction(data[:,:-1], 
                                outliers_ratio=outliers_ratio, method=method, 
                                coreset_size=top_num, all_outliers=False,
                                y = data_y,loss_function=loss_function,is_kmeans=is_kmeans,
                                initial_centers=theta_tuta,problem_attributes=problem_attributes)   
            end_time = time.time()
            original_time += end_time - start_time
        original_time = original_time/n_runs_original
        
        
        id = get_uuid()
        filename = problem +dataset + id +".pkl"
        path = os.path.join(dir_name,filename)
        
            #shutil.rmtree(dir_name, ignore_errors=True) 
        

        ans = {(dataset,n_level,method):[original_time/ti, original_time/td, original_time/tu]}
        print(ans)
        with open(path, 'wb') as f:
            pickle.dump(ans, f)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        return T,binary_tree
    return    

    



# %%

if __name__ == "__main__":
    X_train_weighted = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

    #cifar_train_data_weighted = np.hstack((cifar_train_data, np.ones((cifar_train_data.shape[0], 1))))
    #datasets1 = {"MNIST":X_train_weighted,"CIFAR-10":cifar_train_data_weighted}
    #datasets2 = {"covtype":covtype_train_data}
    #datasets3 = {"USCensus":census_train_data}
    n_levels_small = [10,9,8,7,6,5,4,3]
    n_levels = [12,11,10,9,8,7,6,5,4,3]
    seeds = range(5)
    methods = ["uniform","GSP","importance","ring"]
    new = ["ring"]
    param= [
            # [("./dyn_logi_co_4","modified_covtype", n_level,0.1,200,5, None, method, logistic_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]+
            
            # ("./dyn_logi_kdd1","KDD", n_level,0.1,200,5, None, method, logistic_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]+[
            # +[
            # ("./dyn_logi_har","HAR", n_level,0.1,200,5, None, method, logistic_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]+[
            # ("./dyn_kmeans_har","HAR", n_level,0.1,200,5,3, method, kmeans_loss, seed) 
            # for method in methods  for n_level in n_levels for seed in seeds]+[
            ("./dyn_kmeans_co","Covertype", n_level,0.1,200,5, 7, method, kmeans_loss, seed)
            for method in new for n_level in n_levels for seed in seeds]+[
            ("./dyn_kmeans_us","USCensus", n_level,0.1,200,5, 5, method, kmeans_loss, seed)
            for method in new for n_level in n_levels for seed in seeds]
    # +[
            # ("./dyn_huber_trip1","Tripfare", n_level,0.1,200,5, 5, method, huber_regression_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]+[
            # ("./dyn_huber_gpu1","GPU", n_level,0.1,200,5, 5, method, huber_regression_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]+[
            # ("./dyn_huber_query1","Query", n_level,0.1,200,5, 5, method, huber_regression_loss, seed)
            # for method in methods for n_level in n_levels for seed in seeds]
            
    # 提前清空
    # for para in param:
    #     dir_name,dataset,n_level,outliers_ratio,n_runs,n_runs_original,problem_attributes, method, loss_function,seed = para
    #     if os.path.exists(dir_name):
    #         shutil.rmtree(dir_name, ignore_errors=True)


    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=ctx,max_workers=10) as executor:
        try:
            futures = [executor.submit(evaluate, _) for _ in param]

            # 获取每个任务的结果
            for future in concurrent.futures.as_completed(futures):
                #result = future.result()
                print(f"Task result received")

            print("All tasks are completed.")
        except:
            print("Error")
    # for _ in param:
    #     try:
    #         evaluate(_)
    #         print("Task result received")
    #     except:
    #         print("Error")
    # evaluate(param[0])



# %%
# n_levels = [3,11,12,13]
# methods = ["uniform","GSP","importance"]

# param1= [("./test","modified_covtype", n_level,0.05,10,5, 20, method, logistic_loss)for method in methods  for n_level in n_levels]
# param = param1
# #print("???")
# profiler = cProfile.Profile()
# profiler.enable()
# T, tree = evaluate(param[0])
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumulative')
# stats.print_stats()

# %%
# n_levels = [3,11,12,13]
# methods = ["uniform","GSP","importance"]

# param1= [("./test","modified_covtype", n_level,0.05,10,5, 20, method, logistic_loss)for method in methods  for n_level in n_levels]
# param = param1
# #print("???")

# T, tree = evaluate(param[0])

