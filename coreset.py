# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn import metrics
import os
import sys
import coresets
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import gzip
import tarfile
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
import uuid
from sklearnex import patch_sklearn
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
import multiprocessing
import traceback
debug = 0
test = 0
inner_parallel_worker = 1
def get_uuid():
    return str(uuid.uuid4())
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

from sklearn.preprocessing import MinMaxScaler


cpu_num = 10 # 这里设置成你想运行的CPU个数
#os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

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


        X_train_indices = np.random.choice(X_.shape[0], int(X_.shape[0] * 0.9), replace=False)
        X_test_indices = np.setdiff1d(np.arange(X_.shape[0]), X_train_indices)

        X = X_[X_train_indices]
        labels = labels_[X_train_indices].flatten()

        Verify_X = X_[X_test_indices]
        Verify_labels = labels_[X_test_indices].flatten()
        
        #print("修改后的标签:", modified_labels)
    elif dataset_name == "Tripfare":
        path = "/home/sjj/Experiment/data/taxi_fare/train.csv"
        data = pd.read_csv(path)
        #data
        data_np = data.to_numpy()
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
def H(z,epsilon=1.35):
    ans = np.where(
        np.abs(z) < epsilon,
        (z**2),
        (2 * epsilon * np.abs(z) - epsilon**2)
    )
    return ans.flatten()
# here problem_attributes is epsilon
def regression_loss(X,y,problem_attributes=None,theta=None,weights=None,z=None):
    if weights is None:
        weights = np.ones(X.shape[0])
    losses = np.empty(X.shape[0])
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
    unweighted_losses = (y.reshape(-1,1)-intercept-np.dot(X,np_w))**2
    #print(losses.sum())
    if z is None:
        losses = weights.reshape(-1,1)*unweighted_losses
        return losses.flatten()
    else:
        #print(unweighted_losses[:10].flatten())
        sorted_indices = np.argsort(unweighted_losses.flatten())
        cumulative_weights = np.cumsum(weights[sorted_indices])
        #remove_index = np.searchsorted(cumulative_weights, z, side='right')
        # 这个才是对的！！
        normal_indices = sorted_indices[np.where(cumulative_weights <= cumulative_weights[-1]-z)[0]]
        
        outlier_indices = np.setdiff1d(sorted_indices, normal_indices)
        #print("####")       
        #print(len(outlier_indices))
        #print("####")
        #print(len(normal_indices))
        #print(normal_indices)
        #print(z)        
        outlier_weight = np.sum(weights[outlier_indices])
        if normal_indices.size > 0: 
            normal_max = np.argmax(unweighted_losses[normal_indices])
            weights_ = weights
            weights_[normal_max] = weights[normal_max]-z+outlier_weight
        #print(weights_)
        #print(unweighted_losses[outlier_indices])
        losses = weights_[normal_indices].reshape(-1,1)*unweighted_losses[normal_indices]
        #print(losses.sum())
        return losses.flatten()

def huber_regression_loss(X,y,z=None,problem_attributes=1.35,theta=None,weights=None):
    #print(X.shape)
    #print(y.shape)
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
            if(losses.sum()<0):
                print(losses.sum())
        if z is not None:
            #print(z)
            sorted_indices = np.argsort(unweighted_loss.flatten())
            cumulative_weights = np.cumsum(weights[sorted_indices])
            normal_indices = sorted_indices[np.where(cumulative_weights <= cumulative_weights[-1]-z)[0]]
            outlier_indices = np.setdiff1d(sorted_indices, normal_indices)
            # print(len(normal_indices))
            # print(weights[outlier_indices])
            # print(weights[normal_indices])
            # print(normal_indices)
            # print(z)
            # print(cumulative_weights[-1]-z)
            #print(weights[sorted_indices])
            outlier_weight = np.sum(weights[outlier_indices])
            weights_ = weights
            if normal_indices.size > 0: 
                normal_max = np.argmax(unweighted_loss[normal_indices])
                weights_[normal_max] = weights[normal_max]-z+outlier_weight

            losses = (weights_[normal_indices].reshape(-1,1))*(unweighted_loss[normal_indices].reshape(-1,1))
            if(losses.sum()<0):
                print(losses.sum())
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
def kmeans_loss(X,y,problem_attributes=None,theta=None,weights=None):
    if theta is None:
        initial_model = KMeans(n_clusters=problem_attributes, init='k-means++', n_init=1, max_iter=1, random_state=0)
        initial_model.fit(X)
        theta = initial_model.cluster_centers_
    if weights is None:
        weights = np.ones(X.shape[0])
    labels = np.argmin(cdist(X, theta), axis=1)
    #losses = np.empty(X.shape[0])
    #print((X - theta[labels])**2.shape)
    losses = weights.reshape(-1,1)*(np.sum((X - theta[labels])**2,axis=1)).reshape(-1,1)
    #for i in range(X.shape[0]):
    #    losses[i] = weights[i]*np.sum((X[i] - theta[labels[i]]) ** 2)
    return losses.flatten()

# %%
def logistic_loss(X,y,problem_attributes=None,theta=None,weights=None):
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
        #print(linear_combination)
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
    except:
        return None


# %% 
def logistic_outlier(X,y,z,max_iter=30,tol=1e-4,initial_theta=None, weights=None,init=False,verify_X=None,verify_y=None):
    # using warmstart,always iter 1 is enough
    y = y.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        logistic = LogisticRegression(warm_start=True,max_iter=1,penalty=None)
        if weights is None:
            weights = np.ones(X.shape[0])
        if init:
            # logistic_init = LogisticRegression(max_iter=5)
            logistic_init = LogisticRegression()
            logistic_init.fit(X,y,weights)
            theta_tuta = np.array(logistic_init.intercept_)
            theta_tuta = np.append(theta_tuta,logistic_init.coef_)
            return theta_tuta
        
        logistic.fit(X,y,weights)
        if initial_theta is not None:
            #print(np.array(initial_theta[0]))
            logistic.intercept_ = np.array(initial_theta[0]).flatten()
            logistic.coef_ = np.array(initial_theta[1:]).reshape(1,-1)
        
        old_theta = np.insert(logistic.coef_,0,logistic.intercept_)
        old_loss = 1
        start = time.time()
        for iter in range(max_iter):
            # 不带权的loss
            losses = logistic_loss(X,y,theta=old_theta)
            sorted_indices = np.argsort(losses.flatten())
            cumulative_weights = np.cumsum(weights[sorted_indices])
            
            normal_indices = sorted_indices[np.where(cumulative_weights <= cumulative_weights[-1]-z)[0]]
            outlier_indices = np.setdiff1d(sorted_indices, normal_indices)

            outlier_weight = np.sum(weights[outlier_indices])
            weights_ = weights
            if normal_indices.size > 0: 
                normal_max = np.argmax(losses[normal_indices])
                weights_ = weights
                weights_[normal_max] = weights[normal_max]-z+outlier_weight
            #print(len(normal_indices)/len(X))
            logistic.fit(X[normal_indices],y[normal_indices],weights_[normal_indices])
            new_theta = np.insert(logistic.coef_,0,logistic.intercept_)
            new_loss = logistic_loss(X[normal_indices],y[normal_indices],theta = new_theta, weights=weights_[normal_indices]).sum()
            relative_loss_change = (new_loss - old_loss)/old_loss
            relative_change = np.linalg.norm(new_theta - old_theta)/np.linalg.norm(old_theta)
            if abs(relative_loss_change) < tol:
                #print("OK!")
                break
            old_theta = new_theta
            old_loss = new_loss
        end = time.time()
        print(iter)
        
        duration = end - start
        #print(len(X))
        #print(len(X[normal_indices]))
        #print(logistic.score(X,y))
        loss = logistic_loss(X,y,theta = new_theta,weights=weights).sum()
        robust_loss = logistic_loss(X[normal_indices],y[normal_indices],theta = new_theta, weights=weights_[normal_indices]).sum()
        if (loss is None):
            print("loss is None")
        elif (robust_loss is None):
            print("rloss is None")
        acc = logistic.score(verify_X,verify_y)
        print(acc)
    return loss,robust_loss,duration,acc
# %%
def huber_regression_outlier(X,y,epsilon,z, max_iter=30,tol=1e-4, initial_theta=None, weights=None,init=False):
    y = y.flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        huber = HuberRegressor(warm_start=True,max_iter=max_iter)
        if weights is None:
            weights = np.ones(X.shape[0])
        if init:
            #huber_init = HuberRegressor(max_iter=1)
            huber_init = HuberRegressor()
            huber_init.fit(X,y,weights)
            theta_tuta = np.array([huber_init.scale_,huber_init.intercept_])
            theta_tuta = np.append(theta_tuta,huber_init.coef_)
            return theta_tuta
        if initial_theta is not None:
            huber.fit(X,y,weights)
            huber.scale_ = initial_theta[0]
            huber.intercept_ = initial_theta[1]
            huber.coef_ = initial_theta[2:]
        
        start = time.time()
        huber.fit(X,y,weights)
        end = time.time()
        #print(huber.n_iter_,len(X))
        duration = end - start
        loss = huber_regression_loss(X,y,z=None,problem_attributes=epsilon,theta=[huber.scale_,huber.intercept_,huber.coef_],weights=weights).sum()
        # if(loss <0):
        #     print("?")
        robust_loss = regression_loss(X,y,z=z,theta=[huber.scale_,huber.intercept_,huber.coef_],weights=weights).sum()
        #robust_loss = huber_regression_loss(X,y,z=z,problem_attributes=epsilon,theta=[huber.scale_,huber.intercept_,huber.coef_],weights=weights).sum()
        # if(robust_loss<0):
        #     print("??")
        #sorted_loss = np.sort(losses)
        #robust_loss = sorted_loss[:-z].sum()
    return loss,robust_loss,duration,None

# %%
def kmeans_outlier(df,y, n_clusters, z, max_iter=10,tol=1e-4, initial_centers=None, weights=None, 
                   verify_dataset=None,init=False):
    #debug = 1
    # 初始化KMeans模型
    if verify_dataset is None:
        verify_dataset = df
    if weights is None:
        weights = np.ones(df.shape[0])
    if (initial_centers is None)|init:

        #kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5, max_iter=1)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5)
        kmeans.fit(df)
        old_centers = kmeans.cluster_centers_
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, max_iter=1)
        old_centers = initial_centers
    y_true = y
    old_loss = 1
    old_robust_loss = 1
    if init:
        return old_centers
    start = time.time()
    for iter in range(max_iter): 
        #print(iter)
        #print(old_centers)    
        #distances = cdist(df, old_centers)
        #min_distances = distances.min(axis=1)
        losses = kmeans_loss(df,y,n_clusters,old_centers)
        # 不用weighted排序
        # weighted_loss = min_distances*min_distances * weights
        # 对距离进行排序，并找到距离最大的前一定比例的数据点
        sorted_indices = np.argsort(losses.flatten())
        cumulative_weights = np.cumsum(weights[sorted_indices])
        normal_indices = sorted_indices[np.where(cumulative_weights <= cumulative_weights[-1]-z)[0]]
        outlier_indices = np.setdiff1d(sorted_indices, normal_indices)
        
        outlier_weight = np.sum(weights[outlier_indices])
        # if test:
        #     print(len(normal_indices),len(outlier_indices),outlier_weight,cumulative_weights[-1])
        weights_ = weights
        if normal_indices.size > 0: 
            normal_max = np.argmax(losses[normal_indices])
            weights_ = weights
            weights_[normal_max] = weights[normal_max]-z+outlier_weight
        

        tmp = KMeans(n_clusters=n_clusters, init=old_centers, max_iter=1)
        tmp.fit(df[normal_indices,:], sample_weight=weights_[normal_indices])
        #new_loss = kmeans_loss(np.array(df),y=y,problem_attributes=n_clusters,weights=weights, theta=tmp.cluster_centers_).sum()
        # loss = kmeans_loss(np.array(df),weights=weights, centers=tmp.cluster_centers_).sum()
        # robust_loss = kmeans_loss(np.array(df[normal_indices,:]), weights=weights[normal_indices], centers=tmp.cluster_centers_).sum()

        relative_change = np.linalg.norm(tmp.cluster_centers_ - old_centers) / np.linalg.norm(old_centers)
        #relative_loss_change = (old_loss - new_loss)/old_loss
        #old_loss = new_loss
        new_loss = kmeans_loss(np.array(df),y=y,problem_attributes=n_clusters,weights=weights, theta=tmp.cluster_centers_).sum()
        new_robust_loss = kmeans_loss(np.array(df[normal_indices,:]),y=y,problem_attributes=n_clusters, weights=weights[normal_indices], theta=tmp.cluster_centers_).sum()
        relative_loss_change = (old_loss-new_loss)/old_loss
        relative_robust_loss_change = (old_robust_loss-new_robust_loss)/old_robust_loss
        #print(relative_loss_change,relative_robust_loss_change)
        if debug:
            print(iter)
            print(new_loss,old_loss)
            print(new_robust_loss,old_robust_loss)
            print(relative_change)
            print(relative_loss_change)
            print(relative_robust_loss_change)
        # 没考虑强收敛！
        if abs(relative_robust_loss_change) < tol:
            if debug:
                #print(f"Loss: {loss}")
                #print(relative_change)
                print("Converged at iteration ", iter)
            break
            print("OK")
        old_centers = np.copy(tmp.cluster_centers_)
        old_robust_loss = new_robust_loss
        old_loss = new_loss
    #print(iter)
    end = time.time()
    duration = end - start
    loss = kmeans_loss(np.array(df),y=y,problem_attributes=n_clusters,weights=weights, theta=tmp.cluster_centers_).sum()
    robust_loss = kmeans_loss(np.array(df[normal_indices,:]),y=y,problem_attributes=n_clusters, weights=weights[normal_indices], theta=tmp.cluster_centers_).sum()
    return loss,robust_loss,duration,None


# %%
# replace = True
def coreset_importance(X,y=None,weights = None,problem_attributes=None, coreset_size=1000,initial_centers=None,loss_function=None,is_kmeans=False):
    # todo implement importance sample
    try:
        simple_version = 1
        if simple_version:
            
            if is_kmeans:
                coreset_gen = coresets.KMeansCoreset(X, init = initial_centers)
                C,w=coreset_gen.generate_coreset(coreset_size)
                coreset = np.hstack((C, w.reshape(-1,1)))
                # ? 这里的y不对
                return coreset, y.reshape(-1,1)
            else:
                # QR-decompositon of D_w X
                print("running qr!!!!!!!!!!!!")
                if weights is None:
                    weights = np.ones(X.shape[0])
                # D_w = np.diag(weights)

                # print(1)
                # X_w = np.dot(D_w,X)
                X_w = weights[:, np.newaxis] * X

                #print(2)
                Q,R = np.linalg.qr(X_w)
                #print(3)
                #print(Q.shape)
                #print(R.shape)
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
                w = weight_vec[sample_index]
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
        return None

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
            print(X.shape)
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
    # 这里无所谓
def coreset_uniform(X, y=None, weights = None, coreset_size=1000):
    X_copy = X.copy()
    n_samples, n_features = X.shape
    indices = np.random.choice(n_samples, size=coreset_size, replace=False)
    sample_weights = np.ones((n_samples,)) * n_samples / coreset_size
    if weights is not None:
        sample_weights = sample_weights * weights[indices]
    coreset = np.hstack((X_copy[indices], sample_weights[indices].reshape(-1, 1)))
    return coreset, y[indices].reshape(-1,1)

# %%
# GSP 这里没改np.random.choice Replace
# 加了weight情况，未测试
# 有点问题
# def coreset_GSP(X, y=None, weights=None,problem_attributes=None, coreset_size = 1000, initial_centers  = None, loss_function = kmeans_loss):
#     try:
       
       
#         X_copy = X.copy()
#         n_samples, n_features = X.shape
#         #centers = initial_centers
#         losses = loss_function(X_copy,y, problem_attributes=problem_attributes, theta=initial_centers)
#         #print(losses)
        
#         T = losses.mean()
#         tau = losses.min()
#         lower = tau
#         upper = T + tau
#         #layer = 0
#         # 这里label指第几层
#         X_with_labels = []
#         Y_with_labels = []
#         weight_vec = []
#         #w_with_labels = []
#         # 这里没维护indices 有点冗余了
#         coreset = np.array([]).reshape(0, n_features + 1)
#         coreset_y = np.array([]).reshape(0, 1)
#         non_empty_layer = 0
#         not_full_arr = [True]*(np.log2(n_samples).astype(int) + 1)
#         indices_arr = [None]*(np.log2(n_samples).astype(int) + 1)
#         min_num = -1
#         min_layer = -1
#         #original_arr = [None]*(np.log2(n_samples).astype(int) + 1)
#         for _ in range(np.log2(n_samples).astype(int) + 1):
#             indices = np.where((losses >= lower) & (losses < upper))[0]
            
#             if len(indices) > 0:
#                 non_empty_layer += 1
#                 indices_arr[_] = indices
#             lower = tau + T * np.power(2,_)
#             upper = tau + T * np.power(2,_+1)
#         samples_per_layer = int(coreset_size / non_empty_layer)
#         #print(np.log2(n_samples).astype(int) + 1)
        
#         for _ in range(np.log2(n_samples).astype(int) + 1):
#             if indices_arr[_] is None:
#                 continue
#             else:
#                 indices = indices_arr[_]
#                 #print(len(indices))
#             #indices = np.where((losses >= lower) & (losses < upper))[0]
#             weight_vec.append(len(indices)/len(X_copy))
#             if len(indices) > 0 :
#                 #label = np.full((len(indices),), layer)
#                 #X_with_labels.append(np.hstack((X_copy[indices], label.reshape(-1, 1))))
#                 #print(X_copy[indices].shape)
#                 #print(y[indices].reshape(-1,1).shape)
#                 #print(label.reshape(-1,1).shape)
#                 # update: sample here
#                 sample_i = int(coreset_size * weight_vec[_])+1
#                 #print(indices.shape)
#                 #print(X.shape)
#                 sample_indice = np.random.choice(indices, size=sample_i)
#                 weight = len(indices)/len(sample_indice)
#                 # hstack sample together with weight
#                 coreset_i = np.hstack((X_copy[sample_indice], np.full((len(sample_indice),), weight).reshape(-1, 1)))
#                 coreset_y_i = y[sample_indice].reshape(-1, 1)
#                 coreset = np.vstack((coreset, coreset_i))
#                 coreset_y = np.vstack((coreset_y, coreset_y_i))
#                 #Y_with_labels.append(np.hstack((y[indices].reshape(-1,1),label.reshape(-1,1))))
#                 #w_with_labels.append(np.hstack((weights[indices].reshape(-1,1),label.reshape(-1,1))))

#                 #ayer += 1
#             lower = tau + T * np.power(2,_)
#             upper = tau + T * np.power(2,_+1)
#         return coreset,coreset_y

        
#         # while coreset_size > 0:
            
        
#         #samples_per_layer = coreset_size / layer
#         # update: 每层按照原始比例sample
#         # float 
#         # samples_per_layer = coreset_size * weight_vec
#         # X_with_labels = np.vstack(X_with_labels)
#         # Y_with_labels = np.vstack(Y_with_labels)
#         # #w_with_labels = np.vstack(w_with_labels)
        
        
#         # #coreset_w = np.array([]).reshape(0,2)
#         # full = []
#         # indices = []
#         # not_full = range(layer)
#         # while True:
#         #     new_full = []
#         #     new_not_full = []
#         #     for i in not_full:
#         #         X_i = X_with_labels[X_with_labels[:, -1] == i,:]
#         #         Y_i = Y_with_labels[Y_with_labels[:, -1] == i,:]
#         #         #w_i = w_with_labels[w_with_labels[:, -1] == i,:]
#         #         if len(X_i) < samples_per_layer:
#         #             new_full.append(i)
#         #             indices.extend(X_with_labels[:, -1] == i)
#         #             coreset = np.vstack((coreset, X_i))
#         #             coreset_y = np.vstack((coreset_y,Y_i))
#         #             #coreset_w = np.vstack((coreset_w,w_i))
#         #         else:
#         #             new_not_full.append(i)
#         #             coreset_indices = np.random.choice(X_i.shape[0], int(samples_per_layer), replace=False)
#         #             original_indices = (X_with_labels[:, -1] == i)[coreset_indices]
#         #             indices.extend(original_indices)
#         #             coreset = np.vstack((coreset, X_i[coreset_indices]))
#         #             coreset_y = np.vstack((coreset_y,Y_i[coreset_indices]))
#         #             #coreset_w = np.vstack((coreset_w,w_i[coreset_indices]))
#         #     if not new_full:
#         #         break
#         #     else:
#         #         samples_per_layer = (coreset_size - len(coreset)) / len(new_not_full)
#         #         not_full = new_not_full
#         #         full = new_full

        
#         # # weight 
#         # for i in range(layer):
#         #     count_X = np.count_nonzero(X_with_labels[:, -1] == i)
#         #     count_C = np.count_nonzero(coreset[:, -1] == i)
#         #     coreset[coreset[:, -1] == i, -1] = count_X / count_C
#         # if weights is not None:
#         #     coreset[indices,-1] = coreset[:,-1]*weights[indices]
#         # return coreset, coreset_y[:,:-1].reshape(-1,1)
#     except Exception as e:
#         print(e)
#         traceback.print_exc()
#         return None

# %%
# 又有问题，这里用的dynamic_copy的版本
# def coreset_GSP(X, y=None, weights=None,problem_attributes=None, coreset_size = 1000, initial_centers  = None, loss_function = kmeans_loss):
#     try:
#         X_copy = X.copy()
#         n_samples, n_features = X.shape
#         #centers = initial_centers
#         losses = loss_function(X_copy,y, problem_attributes=problem_attributes, theta=initial_centers)
#         #print(losses)
        
#         T = losses.mean()
#         tau = losses.min()
#         lower = tau
#         upper = T + tau
#         layer = 0
#         # 这里label指第几层
#         X_with_labels = []
#         Y_with_labels = []
#         #w_with_labels = []
#         # 这里没维护indices 有点冗余了
#         for _ in range(np.log2(n_samples).astype(int) + 1):
#             indices = np.where((losses >= lower) & (losses < upper))[0]
#             if len(indices) > 0 :
#                 label = np.full((len(indices),), layer)
#                 X_with_labels.append(np.hstack((X_copy[indices], label.reshape(-1, 1))))
#                 #print(X_copy[indices].shape)
#                 #print(y[indices].reshape(-1,1).shape)
#                 #print(label.reshape(-1,1).shape)
#                 Y_with_labels.append(np.hstack((y[indices].reshape(-1,1),label.reshape(-1,1))))
#                 #w_with_labels.append(np.hstack((weights[indices].reshape(-1,1),label.reshape(-1,1))))
#                 layer += 1
#             lower = tau + T * np.power(2,_)
#             upper = tau + T * np.power(2,_+1)
#         samples_per_layer = coreset_size / layer
#         X_with_labels = np.vstack(X_with_labels)
#         Y_with_labels = np.vstack(Y_with_labels)
#         #w_with_labels = np.vstack(w_with_labels)
        
#         coreset = np.array([]).reshape(0, n_features + 1)
#         coreset_y = np.array([]).reshape(0, 2)
#         #coreset_w = np.array([]).reshape(0,2)
#         full = []
#         indices = []
#         not_full = range(layer)
#         while True:
#             new_full = []
#             new_not_full = []
#             for i in not_full:
#                 X_i = X_with_labels[X_with_labels[:, -1] == i,:]
#                 Y_i = Y_with_labels[Y_with_labels[:, -1] == i,:]
#                 #w_i = w_with_labels[w_with_labels[:, -1] == i,:]
#                 if len(X_i) < samples_per_layer:
#                     new_full.append(i)
#                     indices.extend(X_with_labels[:, -1] == i)
#                     coreset = np.vstack((coreset, X_i))
#                     coreset_y = np.vstack((coreset_y,Y_i))
#                     #coreset_w = np.vstack((coreset_w,w_i))
#                 else:
#                     new_not_full.append(i)
#                     coreset_indices = np.random.choice(X_i.shape[0], int(samples_per_layer))
#                     original_indices = (X_with_labels[:, -1] == i)[coreset_indices]
#                     indices.extend(original_indices)
#                     coreset = np.vstack((coreset, X_i[coreset_indices]))
#                     coreset_y = np.vstack((coreset_y,Y_i[coreset_indices]))
#                     #coreset_w = np.vstack((coreset_w,w_i[coreset_indices]))
#             if not new_full:
#                 break
#             else:
#                 samples_per_layer = (coreset_size - len(coreset)) / len(new_not_full)
#                 not_full = new_not_full
#                 full = new_full

        
#         # weight 
#         for i in range(layer):
#             count_X = np.count_nonzero(X_with_labels[:, -1] == i)
#             count_C = np.count_nonzero(coreset[:, -1] == i)
#             coreset[coreset[:, -1] == i, -1] = count_X / count_C
#         # ? 这里有疑问
#         if weights is not None:
#             coreset[:,-1] = coreset[:,-1]*weights[indices]
#         return coreset, coreset_y[:,:-1].reshape(-1,1)
#     except Exception as e:
#         print(e)
#         return None

# %%
def coreset_GSP(X, y=None, weights=None,problem_attributes=None, coreset_size = 1000, initial_centers  = None, loss_function = kmeans_loss):
    try:
        X_copy = X.copy()
        n_samples, n_features = X.shape
        #centers = initial_centers
        losses = loss_function(X_copy,y, problem_attributes=problem_attributes, theta=initial_centers)
        #print(losses)
        
        T = losses.mean()
        tau = losses.min()
        lower = tau
        upper = T + tau
        layer = 0
        # 这里label指第几层
        X_with_labels = []
        Y_with_labels = []
        length_layer = [0]*(np.log2(n_samples).astype(int)+1)
        # 这里没维护indices 有点冗余了
        for _ in range(np.log2(n_samples).astype(int) + 1):
            indices = np.where((losses >= lower) & (losses < upper))[0]
            
            if len(indices) > 0 :
                label = np.full((len(indices),), _)
                X_with_labels.append(np.hstack((X_copy[indices], label.reshape(-1, 1))))
               
                Y_with_labels.append(np.hstack((y[indices].reshape(-1,1),label.reshape(-1,1))))
                length_layer[_] = len(indices)
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
        samples_layer = [0]*(np.log2(n_samples).astype(int)+1)
        not_full = range(np.log2(n_samples).astype(int) + 1)
        while True:
            new_full = []
            new_not_full = []
            size_coreset_i = 0
            for i in not_full:
                if length_layer[i] == 0:
                    continue
                # X_i = X_with_labels[X_with_labels[:, -1] == i,:]
                # Y_i = Y_with_labels[Y_with_labels[:, -1] == i,:]
                #w_i = w_with_labels[w_with_labels[:, -1] == i,:]
                
                if length_layer[i]-samples_layer[i] < samples_per_layer:
                    size_coreset_i+= length_layer[i]-samples_layer[i]
                    samples_layer[i] = length_layer[i]                   
                    
                    new_full.append(i)
                    
                    # indices.extend(np.where(X_with_labels[:, -1] == i)[0])
                    # coreset = np.vstack((coreset, X_i))
                    # coreset_y = np.vstack((coreset_y,Y_i))

                else:
                    size_coreset_i += int(samples_per_layer)
                    samples_layer[i] += int(samples_per_layer)
                    new_not_full.append(i)    
                    
                    # coreset_indices = np.random.choice(X_i.shape[0], int(samples_per_layer))
                    # original_indices = (np.where(X_with_labels[:, -1] == i)[0])[coreset_indices]
                    # indices.extend(original_indices)
                    # coreset = np.vstack((coreset, X_i[coreset_indices]))
                    # coreset_y = np.vstack((coreset_y,Y_i[coreset_indices]))
            if not new_full:
                break
            else:
                samples_per_layer = (coreset_size - size_coreset_i) / len(new_not_full)
                not_full = new_not_full
                full = new_full
        if test:
            print(length_layer)
            print(samples_layer)
        for i in range(layer):
            if samples_layer[i] == 0:
                continue
            X_i = X_with_labels[X_with_labels[:, -1] == i,:]
            Y_i = Y_with_labels[Y_with_labels[:, -1] == i,:]
            if samples_layer[i] == length_layer[i]:
                indices.extend(np.where(X_with_labels[:, -1] == i)[0])
                coreset = np.vstack((coreset, X_i))
                coreset_y = np.vstack((coreset_y,Y_i))
            else:
                coreset_indices = np.random.choice(X_i.shape[0], samples_layer[i])
                original_indices = (np.where(X_with_labels[:, -1] == i)[0])[coreset_indices]
                indices.extend(original_indices)
                coreset = np.vstack((coreset, X_i[coreset_indices]))
                coreset_y = np.vstack((coreset_y,Y_i[coreset_indices]))
        # weight 
        for i in range(layer):
            count_X = np.count_nonzero(X_with_labels[:, -1] == i)
            count_C = np.count_nonzero(coreset[:, -1] == i)
            if count_C == 0:
                continue
            coreset[coreset[:, -1] == i, -1] = count_X / count_C
        # ? 这里有疑问
        # print(coreset.shape)
        # print(len(indices))
        # print(indices)
        # print(weights.shape)
        #print(weights[indices].shape)
        if weights is not None:
            coreset[:,-1] = coreset[:,-1]*weights[indices]
        return coreset, coreset_y[:,:-1].reshape(-1,1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None


# %%
def coreset_construction(X, initial_centers, problem_attributes=None, y=None, outliers_ratio=0.1, 
                         method='uniform', coreset_size=1000, all_outliers=False,loss_function = kmeans_loss,
                         is_kmeans=False, ring_thresh_hold = None):
    X_copy = X.copy()
    y_copy = y.copy()
    n_samples, n_features = X.shape
    #print(method)
    # 计算每个样本属于每个组件的概率
    # 根据outliers_ratio分割样本
    #print(method)
    if method == "all_ring":
        C = coreset_ring(X_copy,y_copy,problem_attributes=problem_attributes,coreset_size=coreset_size,initial_centers=initial_centers,loss_function=loss_function,
                         thresh_hold = ring_thresh_hold)
        C_x, C_y = C
        return C_x, C_y.flatten()
    if method == "all_uniform":
        C = coreset_uniform(X_copy, y=y_copy,coreset_size=coreset_size)
        C_x, C_y = C
        return C_x, C_y.flatten()
    if method == "all_GSP":
        C = coreset_GSP(X_copy, y=y_copy, problem_attributes = problem_attributes, coreset_size=coreset_size, initial_centers=initial_centers,loss_function=loss_function)
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
    #print(outliers_ratio)
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
        C_si = coreset_GSP(X_si, y=Y_si, coreset_size=n_C_si, problem_attributes=problem_attributes, initial_centers=initial_centers,loss_function=loss_function)
    elif method == "importance":
        C_si = coreset_importance(X_si, y=Y_si,coreset_size=n_C_si, problem_attributes=problem_attributes, initial_centers=initial_centers,loss_function=loss_function,
                                  is_kmeans=is_kmeans)
    elif method == "ring":
        C_si = coreset_ring(X_si, y=Y_si, coreset_size=n_C_si, problem_attributes=problem_attributes, initial_centers=initial_centers,loss_function=loss_function)
    #print(5)
    C_si_x,C_si_y = C_si
    C_so_x,C_so_y = C_so
    #print(6)
    #print(C_si_x.shape)
    #print(C_so_x.shape)
    #print(C_si_y.shape)
    #print(C_so_y.shape)
    C_x = np.vstack((C_si_x,C_so_x))
    
    #rint(7)
    C_y = np.vstack((C_si_y,C_so_y))
    #print(4)
    #print('###############')
    #print(C_x.shape,C_y.shape)
    #weights = C_x[:,-1]
    #print(weights[weights<0])
    return C_x,C_y.flatten()







# %%
class Coreset_eval:
    def __init__(self, problem, problem_attributes, problem_outliers_ratio, dataset_name, coreset_attributes, dirpath, 
                nruns = 3,max_iter = 30, noise_mean_var = 0,
                # methods = ["all_GSP", "all_uniform" ,"GSP","uniform","importance","all_importance"]):
                #methods = ["GSP","all_GSP","uniform", "importance"]):
                # methods = ["ring", "uniform" ,"GSP","importance","all_ring","all_uniform","all_GSP","all_importance"]):
                # methods = ["all_ring","all_uniform","all_GSP","all_importance","ring", "uniform" ,"GSP","importance"]):
                # methods = ["ring", "uniform" ,"GSP","importance"]):
                # methods = ["ring","all_ring"]):
                # methods = ["uniform","all_uniform"]):
                # methods = ["all_ring","all_uniform","all_GSP","all_importance"]):
                methods = ["all_ring", "ring", "all_uniform", "uniform", "all_GSP", "GSP", "all_importance", "importance"]):
        np.random.seed(0)
        # updated add noise_mean_var for logistic
        self.noise_mean_var= noise_mean_var
        self.problem = problem
        
        # i.e. n_clusters for k_means
        self.problem_attributes = problem_attributes
        self.problem_outliers_ratio = problem_outliers_ratio
        self.dataset_name = dataset_name
        self.X, self.labels, self.Verify_X, self.Verify_y = load_data(dataset_name)
        
        if (self.problem !="k_means"):
            print("Scaler!!")
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            if (self.problem =="logistic"):
               #? 这里有点小bug 应该用同一个scaler
               self.Verify_X = scaler.fit_transform(self.Verify_X)
            
        else:
            #scaler = StandardScaler(with_std=False)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            # scaler = MinMaxScaler(feature_range=(-20, 20))
            self.X = scaler.fit_transform(self.X)
            #pass
        if (self.problem == "huber_regression"):
            scaler = StandardScaler()
            self.labels = scaler.fit_transform(self.labels.reshape(-1,1)).flatten() 
        self.coreset_sizes, self.noise_ratio, self.noise_var, self.outliers_ratio , self.ring_thresh_hold = coreset_attributes
        # if os.path.exists(dirpath):
        #     shutil.rmtree(dirpath, ignore_errors=True) 
        #     pass
        # os.makedirs(dirpath)
        self.dirpath = dirpath
        self.nruns = nruns
        self.max_iter = max_iter
        self.methods = methods
        
        self.z = int(len(self.X) * self.problem_outliers_ratio/(1-self.noise_ratio))
        self.theta_tuta = None
        if self.problem == "k_means":
            self.loss_function = kmeans_loss
        if self.problem == "huber_regression":
            self.loss_function = regression_loss
        if self.problem == "logistic":
            self.loss_function = logistic_loss
            self.z = int(len(self.X)*self.problem_outliers_ratio)
    def problem_blackbox(self,X,labels,weights = None,init = False,seed = 0):
        #np.random.seed(seed)
        iter = self.max_iter
        X_copy = X.copy()
        y_copy = labels.copy()
        if init:
            #iter = 1
            # print(X.shape[0])
            # print(y_copy.shape)
            # print(X_copy.shape)
            indices = np.random.choice(X.shape[0], size=int(0.001*X.shape[0]), replace=False)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
        if self.problem == "k_means":
            #z = int(self.outliers_ratio*len(X))
            
            res = kmeans_outlier(X_copy,y=y_copy, n_clusters=self.problem_attributes, initial_centers=self.theta_tuta,
                                 z = self.z, max_iter=iter,weights=weights,init=init)
            if init:
                self.theta_tuta = res
                #print(res)
                return
            return res
        if self.problem == "huber_regression":
            #TODO 调研下怎么改huber_regression的比例

            res = huber_regression_outlier(X_copy,y=y_copy,epsilon=1.35,z=self.z,
                                           max_iter=iter,weights=weights,initial_theta=self.theta_tuta,init=init)
            if init:
                self.theta_tuta = res
            return res
        if self.problem == "logistic":

            res = logistic_outlier(X=X_copy,y=y_copy,z=self.z,
                                   max_iter=iter,weights=weights,initial_theta=self.theta_tuta,
                                   init=init,verify_X=self.Verify_X,verify_y=self.Verify_y)
            if init:
                self.theta_tuta = res
            return res
    def parallel_evaluation_coresets(self,args):
        # 待优化

        try:
            seed,dirpath,noise_ratio,noise_var,loss_0,robust_loss_0,original_time,acc_0,dataset_name, data, truth, n_clusters, max_iter, outliers_ratio, initial_centers, method, coreset_size, all_outliers = args
            n_samples, n_features = data.shape
            #print(method)
            if (all_outliers ==True)&(n_samples * outliers_ratio > coreset_size):
                #oversize
                #print("?")
                return
            start_time = time.time()
            #print("01")
            np.random.seed(seed)
            is_kmeans = (self.problem == "k_means")

            C_x,C_y = coreset_construction(X=data, y=truth, problem_attributes=self.problem_attributes, initial_centers=self.theta_tuta, outliers_ratio=outliers_ratio, 
                        method=method, coreset_size=coreset_size,all_outliers=all_outliers,loss_function=self.loss_function,is_kmeans=is_kmeans,ring_thresh_hold=self.ring_thresh_hold)
            end_time = time.time()
            #print(len(C_x))
            #print(len(data))
            t1 = end_time - start_time
            #print(f"Time for coreset construction: {t1}")
            #print("02")
            sys.stdout.flush()
            #print(C_x[:10,-1])
            loss_1,robust_loss_1,t2,acc_1= self.problem_blackbox(X=C_x[:,:-1],labels=C_y, weights=C_x[:,-1])
            #print(f"Time for coreset training: {t2}")
            #print("03")
            sys.stdout.flush()
            coreset_time = t1 + t2
            
            loss_ratio = max(loss_1/loss_0,loss_0/loss_1)
            robust_loss_ratio = max(robust_loss_1/robust_loss_0,robust_loss_0/robust_loss_1)
            time_ratio = original_time / coreset_time
            # print(loss_1,loss_0)
            # print(robust_loss_1,robust_loss_0)
            if test:
                print(loss_ratio,robust_loss_ratio)
            # print(time_ratio)
            acc_ratio = None
            if acc_0 is not None:
                acc_ratio = acc_1/acc_0
            ans = {}
            ans[(self.problem, dataset_name, n_clusters, method, coreset_size, noise_ratio, noise_var, outliers_ratio, all_outliers)] = [loss_ratio, robust_loss_ratio, time_ratio, acc_ratio]
            id = get_uuid()
            filename = self.problem+dataset_name+id+".pkl"
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                print("file exists!!!!!!!")
            sys.stdout.flush()
            
            with open(path, 'wb') as f:
                #print(ans)
                pickle.dump(ans, f)
            
            return ans
        except:
            return None
    def parallel_evaluation(self):        

        #for _ in range(self.nruns):
            # add noise
        np.random.seed(0)
        # todo 能优化..但瓶颈好像在cpu这了
        data = self.X
        truth = self.labels
        n_samples, n_features = data.shape
        if self.noise_ratio > 0:
            if self.problem == "logistic":
                n_noise = int(len(self.X) * self.noise_ratio)
                # update add mean_var
                mean = np.random.normal(0,self.noise_mean_var,(n_noise,n_features))
                #noise = np.random.normal(0, self.noise_var, (n_noise, n_features)) 
                noise = np.random.normal(mean, self.noise_var, (n_noise, n_features)) 
                indices = np.random.choice(data.shape[0],size = n_noise)
                data[indices] += noise
                #truth[indices] = np.random.randint(2,size=n_noise)
                truth[indices] = 1 - truth[indices]
            elif self.problem == "k_means":
                n_noise = int(len(self.X) * self.noise_ratio/(1-self.noise_ratio))
                # mean = np.random.choice([0.5,-0.5],size=(n_noise, n_features))
                mean = np.random.choice([1,-1],size=(n_noise, n_features))
                noise = np.random.normal(mean, self.noise_var, (n_noise, n_features))
                data = np.vstack((data, noise))
            else:
                n_noise = int(len(self.X) * self.noise_ratio/(1-self.noise_ratio))
                mean = np.random.choice([3,-3],size=(n_noise, n_features))
                #print(mean.shape)
                noise = np.random.normal(mean, self.noise_var, (n_noise, n_features))
                #print("?")
                data = np.vstack((data, noise))
            #if self.problem == "logistic":
            #    truth = np.hstack((truth,np.random.randint(2,size=n_noise)))
            if self.problem == "huber_regression":
                mean = np.random.choice([3,-3],size=n_noise)
                truth = np.hstack((truth,np.random.normal(mean,self.noise_var,(n_noise,))))
            if self.problem == "k_means":
                truth = np.hstack((truth, np.full((n_noise,), -1)))
        n_samples, n_features = data.shape
        initial_model = self.problem_blackbox(X = data, labels= truth,init=True)
        #initial_model = KMeans(n_clusters=self.problem_attributes, init='k-means++', n_init=1, max_iter=1, random_state=0)
        #initial_model.fit(data)
        #self.theta_tuta = initial_model.cluster_centers_
        #print(2)
        #print("initialize!")
        #print((self.theta_tuta).shape)
        loss_0, robust_loss_0,original_time,acc=self.problem_blackbox(X = data, labels= truth)
        #print(3)
        sys.stdout.flush()
        #print(self.methods)
        seed_list = [_ +25 for _ in range(self.nruns)]
        # all_outliers_values = [True, False],此处参数表中默认为负, Initial centers 设为None
        coreset_param = [(seed,self.dirpath,self.noise_ratio,self.noise_var,loss_0,robust_loss_0,original_time,acc,self.dataset_name,data,truth,
                        self.problem_attributes,self.max_iter,self.outliers_ratio, None, method, coreset_size, False) 
                        for method in self.methods for coreset_size in self.coreset_sizes for seed in seed_list]
        if test:
            self.parallel_evaluation_coresets(coreset_param[0])
        #print(4)
        else:
        # 下面三行work 尝试改进读写效率
            ctx = multiprocessing.get_context("spawn")  # or "forkserver" but not "fork"
            if self.problem == "k_means":
                inner_parallel_worker = 10
            elif self.problem == "logistic":
                inner_parallel_worker = 5
            elif self.problem == "huber_regression":
                inner_parallel_worker = 1
            with ProcessPoolExecutor(mp_context=ctx,max_workers=inner_parallel_worker) as executor:
                try:
                    #print("?")
                    futures = [executor.submit(self.parallel_evaluation_coresets, param) for param in coreset_param]
                    #executor.map(self.parallel_evaluation_coresets, coreset_param)
                except:
                    print("Error!")
            
          

# %%
def class_parallel(args):
    problem, problem_attributes, problem_outliers_ratio, dataset_name, coreset_attributes, dirpath, nruns, max_iter = args
    
    eval = Coreset_eval(problem=problem, problem_attributes= problem_attributes, problem_outliers_ratio=problem_outliers_ratio, dataset_name=dataset_name, coreset_attributes= coreset_attributes,
                            dirpath=dirpath,nruns=nruns,max_iter=max_iter)
    eval.parallel_evaluation()
    return

# %% 

# %%
    
# %%
### MNIST
### Cifar-10
### Covertype
### modified_covtype
### Energy
### Tripfare

###########################
#参数说明：
#（ problem, problem_parameter, blackbox_problem_solver_parameter, 
# dataset_name, (size, noise_ratio,noise_var,outliers_ratio), output_dir, nruns, max_iter)
# 还有一个比例在coreset_construction 内部，还没写接口
# noise相关：
# shuffle？ reverse？inner_parallel_worker
# GSP 每层layer？

if __name__ == "__main__":

    coreset_sizes_medium = list(range(5000, 21000, 1000))
    coreset_sizes_small = list(range(1000,10000,1000))
    coreset_sizes_small_test = list(range(1000,6000,1000))
    coreset_sizes_small_fortrip = list(range(2000,10000,1000))
    coreset_sizes_tiny = list(range(500,2100,100))
    coreset_sizes_big = list(range(50000,150000,10000))
    param = [

       
            # # 这个不错
            # ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,5,0.005),"./test_kmeans_mn_55",50,100),
           

             # 
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,1.5,0.005),"./test_kmeans_co_1",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,3,0.005),"./test_kmeans_co_2",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,5,0.005),"./test_kmeans_co_3",50,100),


            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.025,1.5,0.005),"./test_kmeans_co_4",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.025,3,0.005),"./test_kmeans_co_5",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.025,5,0.005),"./test_kmeans_co_6",50,100),

            #  ("k_means",7, 0.1,"Covertype", (coreset_sizes_small,0.05,1.5,0.005),"./test_kmeans_co_7",50,100),
            # ("k_means",7, 0.1,"Covertype", (coreset_sizes_small,0.05,3,0.005),"./test_kmeans_co_8",50,100),
            # ("k_means",7, 0.1,"Covertype", (coreset_sizes_small,0.05,5,0.005),"./test_kmeans_co_9",50,100),


            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.005),"./test_kmeans_co_10",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1,0.005),"./test_kmeans_co_11",50,100),
            
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.001),"./test_kmeans_co_12",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1,0.001),"./test_kmeans_co_13",50,100),
           
        #    ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.01),"./test_kmeans_co_14",50,100),
        #     ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1,0.01),"./test_kmeans_co_15",50,100),
           
           #("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.01,0.5,0.001),"./test_kmeans_co_16",50,100),
            # goat
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,0.1,0.01),"./test_kmeans_co_17",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,0.1,0.005),"./test_kmeans_co_18",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,1,0.01),"./test_kmeans_co_19",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,1,0.005),"./test_kmeans_co_20",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,3,0.01),"./test_kmeans_co_21",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,3,0.005),"./test_kmeans_co_22",50,100),
            
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,0.1,0.01),"./test_kmeans_co_23",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,0.1,0.005),"./test_kmeans_co_24",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,1,0.01),"./test_kmeans_co_25",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,1,0.005),"./test_kmeans_co_26",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,3,0.01),"./test_kmeans_co_27",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.01,3,0.005),"./test_kmeans_co_28",50,100),

            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,0.5,0.01),"./test_kmeans_us_2",50,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,1,0.01),"./test_kmeans_us_3",50,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,0.5,0.025),"./test_kmeans_us_4",50,20),
            # # goat
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,1,0.025),"./test_kmeans_us_5",50,20),

            # goat
            #("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,0.5,0.01),"./test_kmeans_us2",50,20),
            

            #("logistic", None, 0.005,"HAR", (coreset_sizes_small,0.025,10,0.01),"./test_logi_har1",50,50),
            #("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,10,0.05),"./test_logi_cov2",50,50),
            #("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.01), "./test_logi_kdd2", 50, 50),

            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,5,0.01),"./test_huber_trip_test",50,300),
            # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.1,5,0.01),"./test_huber_query1_test",50,100),
            # #可以
            #("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.05,0.1,0.025),"./test_huber_gpu1_test",50,300),

            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.05,3,0.01),"./test_huber_trip_test2",50,300),
            # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.05,3,0.01),"./test_huber_query1_test2",50,100),
            # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.05,3,0.05),"./test_huber_gpu1_test2",50,300),

            
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,5,0.001),"./test_huber_trip_test3",50,300),
            # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.1,5,0.001),"./test_huber_query1_test3",50,100),
            # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.1,5,0.005),"./test_huber_gpu1_test3",50,300),

            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.05,3,0.001),"./test_huber_trip_test4",50,300),
            # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.05,3,0.001),"./test_huber_query1_test4",50,100),
            # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.05,3,0.005),"./test_huber_gpu1_test4",50,300),

            
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.01),"./test_huber_trip_test5",50,300),
            # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.1,10,0.01),"./test_huber_query1_test5",50,100),
            # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.1,10,0.05),"./test_huber_gpu1_test5",50,300),

            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.05,10,0.01),"./test_huber_trip_test6",50,300),
            # # 还行
            #("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.05,0.1,0.05),"./test_huber_query1_test6",50,100),
            # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.05,10,0.05),"./test_huber_gpu1_test6",50,300),

            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,0.5,0.05),"./test_huber_trip_test10",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,0.5,0.05),"./test_huber_trip_test11",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,0.5,0.1),"./test_huber_trip_test12",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,0.5,0.025),"./test_huber_trip_test13",50,300),
            # 可以
            #("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.1,0.1,0.05),"./test_huber_trip_test14",50,300),
        
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.001),"./test_huber_trip_test15",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.0025),"./test_huber_trip_test6",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.005),"./test_huber_trip_test17",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.01),"./test_huber_trip_test18",50,300),
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small_fortrip,0.1,10,0.05),"./test_huber_trip_test19",50,300),
            
            #("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.1),"./test_kmeans_co1",50,100),
            #("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.05),"./test_kmeans_co4",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.01),"./test_kmeans_co2",50,100),
           # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.05,5,0.01),"./test_kmeans_us8",50,20),
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.05,5,0.01),"./test_kmeans_us9",50,20),
            
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.05,5,0.01),"./test_kmeans_us9_2",50,20),
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.05,5,0.005),"./test_kmeans_us9_3",50,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.025,5,0.01),"./test_kmeans_us10",50,20),
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.025,5,0.01),"./test_kmeans_us11",50,20),
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.025,5,0.01),"./test_kmeans_us12",50,20),

            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,5,0.025),"./test_kmeans_us13",50,20),
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.05,5,0.025),"./test_kmeans_us14",50,20),
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.05,5,0.025),"./test_kmeans_us15",50,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.025,5,0.025),"./test_kmeans_us16",50,20),
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.025,5,0.025),"./test_kmeans_us17",50,20),
            # ("k_means",5, 0.05,"USCensus", (coreset_sizes_small,0.025,5,0.025),"./test_kmeans_us18",50,20),
            
            #("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.05,10,0.05),"./test_kmeans_us4_really2",50,20),
            
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.1),"./test_kmeans_co3_1",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,20,0.1),"./test_kmeans_co3_2",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,5,0.1),"./test_kmeans_co3_3",50,100),

            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.075),"./test_kmeans_co3_4",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,20,0.075),"./test_kmeans_co3_5",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,5,0.075),"./test_kmeans_co3_6",50,100),
            #"k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.075),"./test_kmeans_co4",50,100),
            #("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.025),"./test_kmeans_co5",50,100),
            #("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.1,10,0.075),"./test_kmeans_co6",50,100),
            #("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.1,10,0.075),"./test_kmeans_co7",50,100),
            #("k_means",7, 0.1,"Covertype", (coreset_sizes_small,0.1,10,0.075),"./test_kmeans_co8",50,100),

            #("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,5,0.075),"./test_kmeans_co9",50,100),
            #("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,5,0.05),"./test_kmeans_co10",50,100),
            #("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,5,0.075),"./test_kmeans_co11",25,100),
            #("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,5,0.025),"./test_kmeans_co12",50,100),
            #  ("logistic", None, 0.0075,"HAR", (coreset_sizes_small,0.05,10,0.1),"./test_logi_har21",50,50),
            #  ("logistic", None, 0.0075,"HAR", (coreset_sizes_small,0.05,10,0.075),"./test_logi_har22",50,50),

            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.1,2,0.05),"./test_logi_har1",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.1,2,0.05),"./test_logi_har2",50,50),
            #  ("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.1,2,0.05),"./test_logi_har3",50,50),
            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.05,2,0.05),"./test_logi_har4",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.05,2,0.05),"./test_logi_har5",50,50),
            #  # sota?
            #  ("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.05,2,0.05),"./test_logi_har6",50,50),

            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.025,2,0.05),"./test_logi_har7",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.025,2,0.05),"./test_logi_har8",50,50),
            #  ("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.025,2,0.05),"./test_logi_har9",50,50),

            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.1,2,0.1),"./test_logi_har10",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.1,2,0.1),"./test_logi_har11",50,50),
            #  ("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.1,2,0.1),"./test_logi_har12",50,50),

            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.05,2,0.1),"./test_logi_har13",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.05,2,0.1),"./test_logi_har14",50,50),
            #  # sota check
            #("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.05,2,0.1),"./test_logi_har15",50,50),
            #("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.05,5,0.1),"./test_logi_har15_2",50,50),

            #  ("logistic", None, 0.05,"HAR", (coreset_sizes_small,0.025,2,0.1),"./test_logi_har16",50,50),
            #  ("logistic", None, 0.025,"HAR", (coreset_sizes_small,0.025,2,0.1),"./test_logi_har17",50,50),
            #  ("logistic", None, 0.01,"HAR", (coreset_sizes_small,0.025,2,0.1),"./test_logi_har18",50,50),
            # 
            #   ("logistic", None, 0.0075,"HAR", (coreset_sizes_small,0.05,10,0.025),"./test_logi_har19",50,50),
            
            #  ("logistic", None, 0.0075,"HAR", (coreset_sizes_small,0.05,10,0.075),"./test_logi_har20",50,50),
             
             
            #("logistic", None, 0.05,"modified_covtype", (coreset_sizes_small,0.1,2,0.05),"./test7.26.3_con2",50,50),
            # ("logistic",None, 0.05,"modified_covtype", (coreset_sizes_small,0.1,5,0.05),"./test7.26.1",50,50),
            # ("logistic", None, 0.05,"modified_covtype", (coreset_sizes_small,0.05,2,0.05),"./test7.26.2_con",50,50),
            
            
            
            # ("logistic", None, 0.2,"modified_covtype", (coreset_sizes_small,0.01,20,0.1),"./test7.26.3_con_really1",25,50),
            # ("logistic", None, 0.1,"modified_covtype", (coreset_sizes_small,0.01,20,0.1),"./test7.26.3_con_really2",25,50),
            # ("logistic", None, 0.01,"modified_covtype", (coreset_sizes_small,0.01,20,0.1),"./test7.26.3_con_really3",25,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,20,0.1),"./test7.26.3_con_really4",25,50),
            

            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,20,0.1),"./test7.26.3_con_really5",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,10,0.1),"./test7.26.3_con_really6",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,2,0.1),"./test7.26.3_con_really7",50,50),
            
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,20,0.05),"./test7.26.3_con_really8",50,50),
            
            # #sota check
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,10,0.05),"./test7.26.3_con_really9",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.01,2,0.05),"./test7.26.3_con_really10",50,50),

            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,20,0.1),"./test7.26.3_con_really11",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,10,0.1),"./test7.26.3_con_really12",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,2,0.1),"./test7.26.3_con_really13",50,50),
            
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,20,0.05),"./test7.26.3_con_really14",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,10,0.05),"./test7.26.3_con_really15",50,50),
            # ("logistic", None, 0.005,"modified_covtype", (coreset_sizes_small,0.05,2,0.05),"./test7.26.3_con_really16",50,50),


            # ("logistic",None, 0.05,"modified_covtype", (coreset_sizes_small,0.05,5,0.05),"./test7.26.4",50,50),
               
            
            #("logistic",None, 0.1,"modified_covtype", (coreset_sizes_small,0.1,5,0.05),"./test7.26.5_con",50,10),
            # ("logistic",None, 0.1,"modified_covtype", (coreset_sizes_small,0.05,2,0.05),"./test7.26.6",50,10),
            # ("logistic",None, 0.1,"modified_covtype", (coreset_sizes_small,0.1,2,0.05),"./test7.26.7",50,10),
            # ("logistic",None, 0.1,"modified_covtype", (coreset_sizes_small,0.05,5,0.05),"./test7.26.8",50,10),
            
            
            
            # ("logistic",None, 0.025,"modified_covtype", (coreset_sizes_small,0.1,5,0.05),"./test7.26.9",50,10),
            # ("logistic",None, 0.025,"modified_covtype", (coreset_sizes_small,0.05,2,0.05),"./test7.26.10",50,10),
            #("logistic",None, 0.025,"modified_covtype", (coreset_sizes_small,0.1,2,0.05),"./test7.26.11_con",50,10),
            
            
            #("logistic",None, 0.025,"modified_covtype", (coreset_sizes_small,0.05,5,0.05),"./test7.26.12",50,10),
            # ("huber_regression", 20, 0.05,"Tripfare", (coreset_sizes_small,0.05,5,0.025),"./testhuber17",50,300), 
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,5,0.025),"./testhuber18",50,300), 
            # ("huber_regression", 5, 0.05,"Tripfare", (coreset_sizes_small,0.05,5,0.025),"./testhuber19",50,300), 
            # ("huber_regression", 1.35, 0.05,"Tripfare", (coreset_sizes_small,0.05,5,0.025),"./testhuber20",50,300), 

            # ("huber_regression", 20, 0.05,"Tripfare", (coreset_sizes_small,0.05,2,0.025),"./testhuber5",50,300), 
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.05,2,0.025),"./testhuber6",50,300), 
            # ("huber_regression", 5, 0.05,"Tripfare", (coreset_sizes_small,0.05,2,0.025),"./testhuber7",50,300), 
            # ("huber_regression", 1.35, 0.05,"Tripfare", (coreset_sizes_small,0.05,2,0.025),"./testhuber8",50,300), 


            # ("huber_regression", 20, 0.05,"Tripfare", (coreset_sizes_small,0.1,2,0.025),"./testhuber9",50,300), 
            # ("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.1,2,0.025),"./testhuber10",50,300), 
            # ("huber_regression", 5, 0.05,"Tripfare", (coreset_sizes_small,0.1,2,0.025),"./testhuber11",50,300), 
            # ("huber_regression", 1.35, 0.05,"Tripfare", (coreset_sizes_small,0.1,2,0.025),"./testhuber12",50,300), 

            # ("huber_regression", 20, 0.05,"Tripfare", (coreset_sizes_small,0.1,5,0.025),"./testhuber13",50,300), 
            # sota check
            #("huber_regression", 10, 0.05,"Tripfare", (coreset_sizes_small,0.1,5,0.025),"./testhuber14_really",50,300), 
            # ("huber_regression", 5, 0.05,"Tripfare", (coreset_sizes_small,0.1,5,0.025),"./testhuber15",50,300), 
            # ("huber_regression", 1.35, 0.05,"Tripfare", (coreset_sizes_small,0.1,5,0.025),"./testhuber16",50,300), 


            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_mn1",50,100),
            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_mn2",50,100),
            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_mn3",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_mn4",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_mn5",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_mn6",50,100),
            # ("k_means",10, 0.025,"MNIST", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_mn7",50,100),
            # ("k_means",10, 0.025,"MNIST", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_mn8",50,100),
            # ("k_means",10, 0.025,"MNIST", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_mn9",50,100),
            
            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_mn10",50,100),
            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,3,0.025),"./test_kmeans_mn11",50,100),
            # ("k_means",10, 0.075,"MNIST", (coreset_sizes_small,0.05,3,0.05),"./test_kmeans_mn12",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_mn13",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,3,0.025),"./test_kmeans_mn14",50,100),
            # ("k_means",10, 0.05,"MNIST", (coreset_sizes_small,0.05,3,0.05),"./test_kmeans_mn15",50,100),
            # ("k_means",10, 0.025,"MNIST", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_mn16",50,100),
            #
            #  ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.05,3,0.1),"./test_kmeans_mn17_really",50,100),
            
            #sota check
        #     ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,3,0.01),"./test_kmeans_mn17_really1",50,100),
        #    ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,3,0.025),"./test_kmeans_mn17_really2",50,100),
        #    ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,3,0.05),"./test_kmeans_mn17_really3",50,100),
        #    ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,3,0.1),"./test_kmeans_mn17_really4",50,100),
            # ("k_means",10, 0.025,"MNIST", (coreset_sizes_small,0.05,3,0.05),"./test_kmeans_mn18",50,100),

            
           
            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_co1",50,100),
            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_co2",50,100),
            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_co3",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_co4",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_co5",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_co6",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,1.5,0.075),"./test_kmeans_co7",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,1.5,0.025),"./test_kmeans_co8",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,1.5,0.05),"./test_kmeans_co9",50,100),
            

            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_co10",50,100),
            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,3,0.025),"./test_kmeans_co11",50,100),
            # ("k_means",7, 0.075,"Covertype", (coreset_sizes_small,0.05,3,0.05),"./test_kmeans_co12",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_co13",50,100),
            # ("k_means",7, 0.05,"Covertype", (coreset_sizes_small,0.05,3,0.025),"./test_kmeans_co14",50,100),
            # ## 
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.01),"./test_kmeans_co15_really1",50,100),
            
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.025),"./test_kmeans_co15_really2",50,100),
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.05),"./test_kmeans_co15_really3",50,100),
            #sota check
            # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.05,10,0.1),"./test_kmeans_co15_really4",50,100),

            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,3,0.075),"./test_kmeans_co16",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,3,0.025),"./test_kmeans_co17",50,100),
            # ("k_means",7, 0.025,"Covertype", (coreset_sizes_small,0.05,3,0.05),"./test_kmeans_co18",50,100),
            
            
            
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.001,3,0.05),"./test_kmeans_us1",25,20),
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.001,3,0.05),"./test_kmeans_us2",25,20),


            # # sota 3/4 check?
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,10,0.05),"./test_kmeans_us3_really",50,20),
            # ("k_means",5, 0.025,"USCensus", (coreset_sizes_small,0.05,10,0.05),"./test_kmeans_us4_really",50,20),

            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.001,3,0.05),"./test_kmeans_us5",25,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.001,3,0.05),"./test_kmeans_us6",25,20),

            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.001,5,0.05),"./test_kmeans_us7",25,20),
            # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.001,5,0.05),"./test_kmeans_us8",25,20),
          
            # sota check
            # ("huber_regression", 20 ,0.05, "Query", (coreset_sizes_tiny,0.1,5,0.01),"./test_huber_query",50,100),
            
            # sota check
            # ("huber_regression", 20 ,0.05, "GPU", (coreset_sizes_small,0.1,5,0.05),"./test_huber_gpu",50,100),
            

           
            # ("logistic", None, 0.075, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd9", 50, 50),
            # ("logistic", None, 0.05, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd10", 50, 50),
            # ("logistic", None, 0.025, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd11", 50, 50),
            #("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd12", 50, 50),
            # ("logistic", None, 0.005, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd13", 50, 50),
            # ("logistic", None, 0.0025, "KDD", (coreset_sizes_small,0.005,1.5,0.05), "./test_logi_kdd14", 50, 50),

            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.005,1.5,0.025), "./test_logi_kdd15", 50, 50),
            # ("logistic", None, 0.005, "KDD", (coreset_sizes_small,0.005,1.5,0.025), "./test_logi_kdd16", 50, 50),
            # ("logistic", None, 0.0025, "KDD", (coreset_sizes_small,0.005,1.5,0.025), "./test_logi_kdd17", 50, 50),

            # ("logistic", None, 0.075, "KDD", (coreset_sizes_small,0.05,2,0.05), "./test_logi_kdd5", 50, 50),
            # ("logistic", None, 0.05, "KDD", (coreset_sizes_small,0.05,2,0.05), "./test_logi_kdd6", 50, 50),
            # ("logistic", None, 0.025, "KDD", (coreset_sizes_small,0.05,2,0.05), "./test_logi_kdd7", 50, 50),
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,2,0.05), "./test_logi_kdd8", 50, 50),


            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.005), "./test_logi_kdd30", 50, 50),
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.005), "./test_logi_kdd31", 50, 50),

            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.01), "./test_logi_kdd32", 50, 50),
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.01), "./test_logi_kdd33", 50, 50),

            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.05), "./test_logi_kdd34", 50, 50),
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.1), "./test_logi_kdd35", 50, 50),
            #("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.025), "./test_logi_kdd36", 50, 50),
            #sota check!!!
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.01), "./test_logi_kdd", 50, 50),
                
            ]
    
    
    param2 = [
        #稳定ok
        # ("huber_regression", 10 ,0.05, "GPU", (coreset_sizes_tiny,0.05,0.1,0.025,None),"./test_huber_gpu1_test",50,300),
        # ("huber_regression", 10 ,0.05, "Query", (coreset_sizes_tiny,0.05,0.1,0.05,None),"./test_huber_query1_test6",50,100),
        
        # ("k_means",10, 0.01,"MNIST", (coreset_sizes_small_test,0.01,2.5,0.0005,None),"./ring1",25,100),
        # 还行
        # ("k_means",10, 0.01,"MNIST", (coreset_sizes_small_test,0.01,5,0.01,None),"./mnring1",25,100),
        # ("k_means", 3, 0.01,"HAR", (coreset_sizes_small_test,0.05,0.75,0.001,None),"./harring2",25,100),
        
        # ("k_means", 3, 0.01,"HAR", (coreset_sizes_small,0.05,1,0.001,None),"./harring3",50,100),
        
        #  ("k_means",3, 0.01,"HAR", (coreset_sizes_small,0.05,1,0.0025,None),"./harring4",50,100),
        #best
        #  ("k_means",3, 0.01,"HAR", (coreset_sizes_small,0.05,1,0.005,None),"./harring5",50,100),
        
        # ("k_means",10, 0.005,"MNIST", (coreset_sizes_small_test,0.01,0.1,0.01,None),"./mnring6",25,100),
        
        #  ("k_means",10, 0.005,"MNIST", (coreset_sizes_small_test,0.01,0.5,0.01,None),"./mnring7",25,100),

        #  ("k_means",10, 0.015,"MNIST", (coreset_sizes_small_test,0.01,1,0.01,None),"./mnring8",25,100),
        
        # ("k_means",10, 0.015,"MNIST", (coreset_sizes_small_test,0.01,0.1,0.01,None),"./mnring9",25,100),
        
        #  ("k_means",10, 0.015,"MNIST", (coreset_sizes_small_test,0.01,0.5,0.01,None),"./mnring10",25,100),
        
        
        
        
        # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,0.1,0.01,None),"./coring",50,100),
        ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,1,0.025,None),"./usring",25,20),
        
         # ("k_means",10, 0.01,"MNIST", (coreset_sizes_small,0.01,5,0.005,None),"./test_kmeans_mn_change",50,100),
        # ("k_means",7, 0.01,"Covertype", (coreset_sizes_small,0.01,0.1,0.01,None),"./test_kmeans_co_change",50,100),
        # ("k_means",5, 0.01,"USCensus", (coreset_sizes_small,0.05,1,0.025,None),"./test_kmeans_us_change",50,20),


        # ("logistic", None, 0.005,"HAR", (coreset_sizes_small,0.025,10,0.01,None),"./test_logi_har1",50,50),
            # ("logistic", None, 0.01, "KDD", (coreset_sizes_small,0.05,0.1,0.01), "./test_logi_kdd2", 50, 50),


            # 不错
            #  ("logistic", None, 0.025,"modified_covtype", (coreset_sizes_small,0.025,0.5,0.01,None),"./test_logi_cov10",50,50),

    ]
    if test:
        class_parallel(param2[0])
    else:
        ctx = multiprocessing.get_context("spawn")
        

        
        with ProcessPoolExecutor(mp_context=ctx,max_workers=2) as executor:
            try:
                futures = [executor.submit(class_parallel, _) for _ in param2]

                # 获取每个任务的结果
                for future in concurrent.futures.as_completed(futures):
                    #result = future.result()
                    print(f"Task result received")

                print("All tasks are completed.")
            except:
                print("Error")

        # with ProcessPoolExecutor(mp_context=ctx,max_workers=3) as executor:
        #     try:
        #         futures = [executor.submit(class_parallel, _) for _ in param2]

        #         # 获取每个任务的结果
        #         for future in concurrent.futures.as_completed(futures):
        #             #result = future.result()
        #             print(f"Task result received")

        #         print("All tasks are completed.")
        #     except:
        #         print("Error")
    # class_parallel(param[0])
# %%
