# %%
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import scipy
# %%
import scipy.io
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing


TEST_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Test.mat'
TEST_B = 'data/BCI_Comp_III_Wads_2004/Subject_B_Test.mat'

TRUE_LABELS_A = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
TRUE_LABELS_B = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'

MATRIX = ['abcdef',
          'ghijkl',
          'mnopqr',
          'stuvwx',
          'yz1234',
          '56789_']

screen = [['A', 'B', 'C', 'D', 'E', 'F'],
          ['G', 'H', 'I', 'J', 'K', 'L'],
          ['M', 'N', 'O', 'P', 'Q', 'R'],
          ['S', 'T', 'U', 'V', 'W', 'X'],
          ['Y', 'Z', '1', '2', '3', '4'],
          ['5', '6', '7', '8', '9', '_']]

print(screen)
print(len(screen))


# %%
def load_dataset(SUBJECT, flag):
    data = scipy.io.loadmat(SUBJECT)

    # print ('Subject A dataa',data)
    Signal = np.float32(data['Signal'])
    # print ('signal',Signal, Signal.shape)

    Flashing = np.float32(data['Flashing'])
    # print ('flashing',Flashing, Flashing.shape)

    StimulusCode = np.float32(data['StimulusCode'])
    # print ('Stimulus COde',StimulusCode,StimulusCode.shape)
    if flag == 0:
        StimulusType = np.float32(data['StimulusType'])
        # print ('Stimulus type',StimulusType,StimulusType.shape)

        Target = data[
            'TargetChar']  # array([ 'EAEVQTDOJG8RBRGONCEDHCTUIDBPUHMEM6OUXOCFOUKWA4VJEFRZROLHYNQDW_EKTLBWXEPOUIKZERYOOTHQI'],4
        # print ('Target char for subjectA',Target)

        return Signal, Flashing, StimulusCode, StimulusType, Target

    else:
        return Signal, Flashing, StimulusCode


# %%
test_Signal_A1, test_Flashing_A1, test_StimulusCode_A1 = load_dataset(TEST_A, 1)
test_char_size = test_Signal_A1.shape[0]
#############################################################################################
test_Signal_B1, test_Flashing_B1, test_StimulusCode_B1 = load_dataset(TEST_B, 1)
# for i in range(0, test_Signal_A1.shape[0]):
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 500))
#     test_Signal_A1[i] = min_max_scaler.fit_transform(test_Signal_A1[i])
# for i in range(0, test_Signal_B1.shape[0]):
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 500))
#     test_Signal_B1[i] = min_max_scaler.fit_transform(test_Signal_B1[i])

# %%
import scipy.signal

# %%
## DOWNSAMPLING THE SIGNAL

secs = test_Signal_A1.shape[1] / 240  # Number of seconds in signal
samps = int(secs * 120)  # Number of samples to downsample

Signal_A = np.zeros([test_Signal_A1.shape[0], samps, 64])
Flashing_A = np.zeros([test_Signal_A1.shape[0], samps])
StimulusCode_A = np.zeros([test_Signal_A1.shape[0], samps])
StimulusType_A=np.zeros([test_Signal_A1.shape[0],samps])

Signal_B = np.zeros([test_Signal_B1.shape[0], samps, 64])
Flashing_B = np.zeros([test_Signal_B1.shape[0], samps])
StimulusCode_B = np.zeros([test_Signal_B1.shape[0], samps])
StimulusType_B=np.zeros([test_Signal_B1.shape[0],samps])


for i in range(0, test_Signal_B1.shape[0]):
    Signal_A[i, :, :] = scipy.signal.resample(test_Signal_A1[i, :, :], int(samps))
    Signal_B[i, :, :] = scipy.signal.resample(test_Signal_B1[i, :, :], int(samps))
    # print (Flashing_A_240[i,:],Flashing_A_240[i,:].shape)
    # 绝对值
    Flashing_A[i, :] = abs(np.round(scipy.signal.resample(test_Flashing_A1[i, :], int(samps))))
    # print (Flashing_A[i,:],Flashing_A[i,:].shape)
    StimulusCode_A[i, :] = abs(np.floor(scipy.signal.resample(test_StimulusCode_A1[i, :], int(samps)))).astype('int8')
    # print (StimulusCode_A[i,:])
    StimulusType_A[i,:] = abs(np.floor(scipy.signal.resample(test_StimulusCode_A1[i,:], int(samps))))
    # print (StimulusType_A[i,:])

    Flashing_B[i, :] = abs(np.round(scipy.signal.resample(test_Flashing_B1[i, :], int(samps))))
    StimulusCode_B[i, :] = abs(np.floor(scipy.signal.resample(test_StimulusCode_B1[i, :], int(samps))))
    StimulusType_B[i,:] = abs(np.floor(scipy.signal.resample(test_StimulusCode_B1[i,:], int(samps))))

# %%
test_Signal_A = Signal_A
test_Signal_B = Signal_B
# %%
test_Flashing_A = Flashing_A
test_Flashing_B = Flashing_B
# %%
test_StimulusCode_B = StimulusCode_B
test_StimulusCode_A = StimulusCode_A
# %%
### DEFINE P300 WIndow size
window = (48 / 2)  # take a window to get no of datapoints corresponding to 600 ms after onset of stimuli
T = int(3 * window)
print(window / 2.0)


# %%

#### CODE TO FORMAT TEST DATA
#Signal：raw数据
#Flashing;闪烁时间点
#StimulusCode:闪烁类别（0~12）
#Target：对应的闪烁是否是目标
def format_test_data(Signal, Flashing, StimulusCode, Target):
    test_char_size = Signal.shape[0]
    responses = np.zeros([test_char_size, 12, 15, T, 64])

    for epoch in range(0, Signal.shape[0]):
        count = 1;
        rowcolcnt = np.zeros(12)
        for n in range(1, Signal.shape[1]):
            # detect location of sample immediately after the stimuli
            if Flashing[epoch, n] == 0 and Flashing[epoch, n - 1] == 1:
                rowcol = int(StimulusCode[epoch, n - 1]) - 1
                # print (Signal[epoch,n:n+window,:].shape)
                # print (rowcolcnt[int(rowcol)])
                responses[epoch, int(rowcol), int(rowcolcnt[int(rowcol)]), :, :] = Signal[epoch,
                                                                                   n - int(window / 2):n + int(
                                                                                       2.5 * window), :]
                #循环计数，寻找Flashing的“1”
                rowcolcnt[rowcol] = rowcolcnt[rowcol] + 1
                # print (rowcolcnt)
        # print (epoch)
    print('Response for all characters：[特征个数 类别个数  时间数据（600ms） 通道数据]', responses.shape)

    #####################################################################################################################
    ### Taking average over 15 instances of the 12 stimuli, comment to check performance and increase the dataset size- TO DO
    testset = np.mean(responses, axis=2)
    # print ('Testset',testset.shape)
    Target = list(Target)
    # target_ohe=np.zeros([len(Target[0]),36])
    # print (Target[0])
    stimulus_indices = []
    for n_char in range(0, len(Target)):  # character epochs

        # print (Target[n_char])
        # vv=np.where(screen==str(Target[0][n_char]))
        # print (vv)
        # [row,col]

        for row in range(0, 6):
            for col in range(0, 6):
                # print (screen[row][col])
                if (Target[n_char]) is (screen[row][col]):
                    ind = [row + 7, col + 1]
                    stimulus_indices.append(ind)
                    # print (ind)
            ##        print ('here',stimulus_indices[n_char])

    # print (stimulus_indices)
    print('Splitting P300 and non-P300 dataset...')
    # iterate over the 2nd dimension of trainset:trainset (train_char_size, 12, 42, 64) and split as train_char_size*2*42*64 and train_char_size*10*42*64

    test_P300_dataset = np.zeros([test_char_size, 2, T, 64])
    test_non_P300_dataset = np.zeros([test_char_size, 10, T, 64])

    for char_epoch in range(0, testset.shape[0]):
        # choose the i,j out of the 2nd dimension of trainset where i,j comes from stimulus_indices[char_epoch]
        ind_1 = stimulus_indices[char_epoch][0]
        ind_2 = stimulus_indices[char_epoch][1]
        # print (ind_1,ind_2)
        l = 0
        for index in range(0, 12):
            if index == ind_1 - 1 or index == ind_2 - 1:
                test_P300_dataset[char_epoch, 0, :, :] = testset[char_epoch, ind_1 - 1, :, :]
                test_P300_dataset[char_epoch, 1, :, :] = testset[char_epoch, ind_2 - 1, :, :]

            else:
                # print ('here')
                # print (index)
                test_non_P300_dataset[char_epoch, l, :, :] = testset[char_epoch, index, :, :]
                # targets_A[char_epoch,index]=0

                l = l + 1

    # print (np.all(P300_dataset[0,0,:,:])==np.all(trainset[0,5,:,:]))
    print(test_P300_dataset.shape)
    print(test_non_P300_dataset.shape)

    return testset, test_P300_dataset, test_non_P300_dataset


testset_A, test_P300_dataset_A, test_non_P300_dataset_A = format_test_data(test_Signal_A, test_Flashing_A,
                                                                           test_StimulusCode_A, TRUE_LABELS_A)
# testset_A=testset_A.reshape([test_char_size*12,T,64])
print('TestsetA', testset_A.shape)
print('\ntestSetB')
testset_B, test_P300_dataset_B, test_non_P300_dataset_B = format_test_data(test_Signal_B, test_Flashing_B,
                                                                           test_StimulusCode_B, TRUE_LABELS_B)
# testset_B=testset_B.reshape([test_char_size*12,T,64])
print('TestsetB', testset_B.shape)

# %%

print('For subject A')

P300_test_A = np.reshape(test_P300_dataset_A, [100 * 2, 72, 64])
P300_test_label_A = np.ones([P300_test_A.shape[0], 1]).astype('int8')
# test_label_A[:,0:2]=1 ; test_label_A[:,2:12]=0
P300_test_label_A = P300_test_label_A

# test_label_A=np.zeros([test_char_size,12])
# test_label_A[:,0:2]=1 ; test_label_A[:,2:12]=0

non_P300_test_A = np.reshape(test_non_P300_dataset_A, [100 * 10, 72, 64])
non_P300_test_label_A = np.zeros([non_P300_test_A.shape[0], 1]).astype('int8')
non_P300_test_label_A = non_P300_test_label_A

# =create_labels(np.reshape(P300_dataset_A,[train_char_size*2,T,64]),np.reshape(targets_A[:,0:2],[train_char_size*2,1]))

# non_P300_train_A,non_P300_hold_A, non_P300_train_label_A, non_P300_hold_label_A=create_subset(np.reshape(non_P300_dataset_A,[train_char_size*10,T,64]),np.reshape(targets_A[:,2:12],[train_char_size*10,1]))
print('-----------------------------------------------------------------------------------------------------------')

print('Test set of P300 samples')
print(P300_test_A.shape, P300_test_label_A.shape)

print('\nTestg set of non-P300 samples')
print(non_P300_test_A.shape, non_P300_test_label_A.shape)

print('-----------------------------------------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------')

# %%
# CHANGE A to B in following cell
print('for test subject B')
P300_test_B = np.reshape(test_P300_dataset_B, [100 * 2, 72, 64])
P300_test_label_B = np.ones([P300_test_B.shape[0], 1]).astype('int8')
# test_label_B[:,0:2]=1 ; test_label_B[:,2:12]=0
P300_test_label_B = P300_test_label_B

# test_label_B=np.zeros([test_char_size,12])
# test_label_B[:,0:2]=1 ; test_label_B[:,2:12]=0

non_P300_test_B = np.reshape(test_non_P300_dataset_B, [100 * 10, 72, 64])
non_P300_test_label_B = np.zeros([non_P300_test_B.shape[0], 1]).astype('int8')
non_P300_test_label_B = non_P300_test_label_B

print('-----------------------------------------------------------------------------------------------------------')

print('Test set of P300 samples')
print(P300_test_B.shape, P300_test_label_B.shape)

print('\nTest set of non-P300 samples')
print(non_P300_test_B.shape, non_P300_test_label_B.shape)

print('-----------------------------------------------------------------------------------------------------------')
print('----------------------------------')


# %%
#单纯的打乱函数
def shuffle(trainIm_rem, trainL_rem):
    NtrainIm_hold = []
    NtrainL_hold = []

    R = random.sample(range(0, trainIm_rem.shape[0]), trainIm_rem.shape[0])

    for k in R:
        # print (k)
        NtrainIm_hold.append(trainIm_rem[k, :, :])
        NtrainL_hold.append(trainL_rem[k, :])

    return np.array(NtrainIm_hold), np.array(NtrainL_hold)


# AGAIN IGNORE THE NAMING CONVENTION WHICH IS BY DEFAULT A

def create_final_testset(P300_hold_A, non_P300_hold_A, P300_hold_label_A, non_P300_hold_label_A):
    #######################################################
    # Combine the dataset:

    h1 = P300_hold_label_A.shape[0] + non_P300_hold_label_A.shape[0]  # 111
    h2 = P300_hold_label_A.shape[0]  # 18

    dataset_A_hold = np.zeros([h1, T, 64])
    dataset_A_hold[0:h2, :, :] = P300_hold_A
    dataset_A_hold[h2:h1:, :] = non_P300_hold_A
    targets_A_hold = np.zeros([h1, 1])
    targets_A_hold[0:h2, :] = 1

    # print(targets_A_hold.shape)

    ### SHUFFLE ABOVE DATASET and LABELS
    print('-------------------------HOLD AND TRAIN DATASET_CNN READY----------------------------------------------')
    dataset_A_hold, targets_A_hold = shuffle(dataset_A_hold, targets_A_hold)
    print(dataset_A_hold.shape, targets_A_hold.shape)
    ################################

    return dataset_A_hold, targets_A_hold


dataset_A_test, targets_A_test = create_final_testset(P300_test_A, non_P300_test_A, P300_test_label_A,
                                                      non_P300_test_label_A)
dataset_B_test, targets_B_test = create_final_testset(P300_test_B, non_P300_test_B, P300_test_label_B,
                                                      non_P300_test_label_B)

# %%
dataset_A_test1 = {};
dataset_A_test1['data'] = dataset_A_test
dataset_B_test1 = {};
dataset_B_test1['data'] = dataset_B_test

targets_A_test1 = {};
targets_A_test1['labels'] = targets_A_test
targets_B_test1 = {};
targets_B_test1['labels'] = targets_B_test

scipy.io.savemat('test_set_A_d_proc.mat', dataset_A_test1)
scipy.io.savemat('test_set_B_d_proc.mat', dataset_B_test1)
scipy.io.savemat('true_labels_A_d_proc.mat', targets_A_test1)
scipy.io.savemat('true_labels_B_d_proc.mat', targets_B_test1)

# %%

# %%
