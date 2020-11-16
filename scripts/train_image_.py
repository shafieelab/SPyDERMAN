import argparse
import csv
import os
import os.path as osp
import statistics
import tqdm
import time
from datetime import datetime
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils.network as network
import helper_utils.loss as loss
import helper_utils.pre_process as prep
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
from helper_utils.data_list_m import ImageList


from helper_utils.logger import Logger
from helper_utils.sampler import ImbalancedDatasetSampler

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=2000, verbose=False, model_path="../results/"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_path = model_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.val_loss_min != val_loss:
                print('Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model, self.model_path)
        self.val_loss_min = val_loss


def image_classification_test(loader, model, test_10crop=False, num_iterations=0, test='test', class_num=2):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader[test][i]) for i in range(1)]
            for i in range(len(loader[test][0])):
                data = [iter_test[j].next() for j in range(1)]
                inputs = [data[j][0] for j in range(1)]
                labels = data[0][1]
                for j in range(1):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(1):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader[test])
            for i in range(len(loader[test])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, raw_outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(raw_outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    # print(len(all_output))
    # print(len(all_label))

    all_output_numpy = all_output.numpy()
    # all_label_numpy= all_label.cpu().numpy()
    _, predict = torch.max(all_output, 1)

    all_values_CSV = []

    predict_numpy = predict.numpy()


    with open(config["logs_path"] + '/_' + str(num_iterations) + '_confidence_values_.csv', mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(['Image_Name', 'Prediction', 'class_0_conf', 'class_1_conf'])

        for value in range(len(all_output_numpy)):
            csv_writer.writerow(
                [all_label[value], predict_numpy[value], all_output_numpy[value][0], all_output_numpy[value][1]])

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    cm = confusion_matrix(all_label, torch.squeeze(predict).float())

    print(cm)
    print(accuracy)

    # with open(config["output_path"] + '/_' + str(num_iterations) + 'accuracy.csv', mode='w') as file:
    #     csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #
    #
    #     csv_writer.writerow(['Accuracy',str(num_iterations),accuracy])

    # with open(config["output_path"] + '/_' + str(num_iterations) + '_CM.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(cm)
    #
    # f.close()

    # exit()
    return accuracy, cm

    # exit()
    # return no


def validation_loss(loader, model, test_10crop=False, data_name='valid_source', num_iterations=0, class_num=2):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader[data_name])
            for i in range(len(loader[data_name])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                # labels = labels.cuda()
                _, raw_outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(raw_outputs)
                if start_test:
                    all_output = outputs.cpu()
                    all_label = labels
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.cpu()), 0)
                    all_label = torch.cat((all_label, labels), 0)
    # _, predict = torch.max(all_output, 1)

    val_loss = nn.CrossEntropyLoss()(all_output, all_label)

    val_loss = val_loss.numpy().item()

    all_output = all_output.float()
    _, predict = torch.max(all_output, 1)

    all_label = all_label.float()
    val_accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_output_numpy = all_output.numpy()
    predict_numpy = predict.numpy()

    if  class_num == 2:
        with open(config["logs_path"] + '/_' + str(num_iterations) + '_confidence_values_.csv', mode='w') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(['Image_Name', 'Prediction', 'class_0_conf', 'class_1_conf'])

            for value in range(len(all_output_numpy)):
                csv_writer.writerow(
                    [all_label[value], predict_numpy[value], all_output_numpy[value][0], all_output_numpy[value][1]])

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        cm = confusion_matrix(all_label, torch.squeeze(predict).float())

        print(cm)
        print(accuracy)

        return val_accuracy, val_loss, accuracy, cm

    # return val_accuracy,val_loss


def train(config):
    now = dt_string.replace(" ", "_").replace(":", "_").replace(".", "_")
    logger = Logger(config["logs_path"] + "tensorboard/" + now)
    model_path = osp.join(config["output_path"], "best_model.pth.tar")
    early_stopping = EarlyStopping(patience=2000, verbose=True, model_path=model_path)

    # temp_acc = 0.0
    # temp_loss = 10000

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    prep_dict["valid_source"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    valid_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"], data="source")
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=False, num_workers=4, drop_last=True) #sampler=ImbalancedDatasetSampler(dsets["source"])

    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"], data="target")
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["valid_source"] = ImageList(open(data_config["valid_source"]["list_path"]).readlines(), \
                                      transform=prep_dict["valid_source"], data="source")
    dset_loaders["valid_source"] = DataLoader(dsets["valid_source"], batch_size=test_bs, \
                                              shuffle=False, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"], data="target")
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    len_train_valid_source = len(dset_loaders["valid_source"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    best_loss_valid = 10000  # total

    best_loss_transfer = 10000  # total
    best_total_loss_numpy_transfer = best_transfer_loss_numpy_transfer = best_classifier_loss_numpy_transfer = 100000

    best_loss_total = 10000  # total
    best_total_loss_numpy_total = best_transfer_loss_numpy_total = best_classifier_loss_numpy_total = 100000

    best_loss3 = 10000  # transfer
    best_total_loss_numpy_acc = best_transfer_loss_numpy_acc = best_classifier_loss_numpy_acc = 100000

    val_accuracy = 0.0
    val_loss = 10.0

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            itr_log = "num_iterations  " + str(i)

            config["out_file"].write(itr_log + "\n")

            config["out_file"].flush()
            print(itr_log)

            train_time_end = time.time()
            test_time_start = time.time()
            base_network.train(False)
            # image_classification_test(dset_loaders, \
            #                                                                            base_network,
            #                                                                            test_10crop=prep_config[
            #                                                                                "test_10crop"],
            #                                                                            num_iterations=i)

            # val_accuracy, val_loss= validation_loss(dset_loaders, \
            #                                                                            base_network,
            #
            #                                                                            )

            val_accuracy, val_loss, best_acc_new, best_cm = validation_loss(dset_loaders, \
                                                                            base_network,

                                                                            num_iterations=i,
                                                                            data_name='valid_source')

            # val_accuracy_target, val_loss_target, best_acc_new_target, best_cm_target = validation_loss(dset_loaders, \
            #                                                                                             base_network,
            #
            #                                                                                             num_iterations=i,
            #                                                                                             data_name='test')

            temp_model = nn.Sequential(base_network)
            if val_loss < best_loss_valid:
                best_loss_valid = val_loss
                best_acc = val_accuracy
                best_model = copy.deepcopy(temp_model)
                torch.save(best_model, osp.join(config["model_path"], "model_" + str(i) + ".pth.tar"))
                best_itr = i

            # torch.save(temp_model, osp.join(config["model_path"], \
            #     "iter_{:05d}_model.pth.tar".format(i)))

        # if i % config["snapshot_interval"] == 0:
        #     torch.save(nn.Sequential(base_network), osp.join(config["model_path"], \
        #         "iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter

        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss

        total_loss.backward()
        optimizer.step()

        transfer_loss_numpy = transfer_loss.clone().cpu().detach().numpy()
        classifier_loss_numpy = classifier_loss.clone().cpu().detach().numpy()
        total_loss_numpy = total_loss.clone().cpu().detach().numpy()
        entropy_numpy = torch.sum(entropy).clone().cpu().detach().numpy()

        info = {'total_loss': total_loss_numpy.item(),
                'classifier_loss': classifier_loss_numpy.item(), 'transfer_loss': transfer_loss_numpy.item(),
                'entropy': entropy_numpy.item(),

                'valid_source_loss': val_loss, 'valid_source_acc': val_accuracy,
                # 'target_valid_loss': val_loss_target, 'target_valid_acc': val_accuracy_target,

                }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, i)

        with open(config["logs_path"] + '/loss_values_.csv', mode='a') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(
                [i, total_loss_numpy, transfer_loss_numpy, classifier_loss_numpy, entropy_numpy, val_loss, val_accuracy,

                 ])

        early_stopping(val_loss, nn.Sequential(base_network))

        # print(i)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # temp_model_total = nn.Sequential(base_network)
        # temp_loss_total = transfer_loss_numpy
        # if temp_loss_total < best_loss_total:
        #     best_loss_total = temp_loss_total
        #     best_model_total = temp_model_total
        #     best_itr_total = i
        #
        #     best_classifier_loss_numpy_total = classifier_loss_numpy
        #     best_total_loss_numpy_total = total_loss_numpy
        #     best_transfer_loss_numpy_total = transfer_loss_numpy
        #
        # temp_model_transfer = nn.Sequential(base_network)
        # temp_loss_transfer = transfer_loss_numpy
        # if temp_loss_transfer < best_loss_transfer:
        #     best_loss_transfer = temp_loss_transfer
        #     best_model_transfer = temp_model_transfer
        #     best_itr_transfer = i
        #
        #     best_classifier_loss_numpy_transfer = classifier_loss_numpy
        #     best_total_loss_numpy_transfer = total_loss_numpy
        #     best_transfer_loss_numpy_transfer = transfer_loss_numpy

    # torch.save(best_model_transfer, osp.join(config["model_path"], "best_model_transfer.pth.tar"))
    # torch.save(best_model_total, osp.join(config["model_path"], "best_model_total.pth.tar"))
    #

    def post_training(model, best_itr, best_classifier_loss, best_total_loss, best_transfer_loss, metric_name):
        model.train(False)
        #

        if is_training:
            torch.save(best_model, osp.join(config["model_path"], "best_model_acc.pth.tar"))

            torch.save(best_model, osp.join(config["model_path"], "best_model" + str(best_itr) + ".pth.tar"))

            model.train(False)
            model = torch.load(osp.join(config["model_path"], "best_model" + str(best_itr) + ".pth.tar"))
        else:
            model = torch.load(model_path_for_testing)
            best_itr = config["best_itr"]

        source_val_accuracy, source_val_loss, best_acc_new, best_cm = validation_loss(dset_loaders, \
                                                                                      model,

                                                                                      num_iterations=best_itr,
                                                                                      data_name='valid_source')

        # val_accuracy_target, val_loss_target, best_acc_new_target, best_cm_target = validation_loss(dset_loaders, \
        #                                                                                             model,
        #
        #                                                                                             num_iterations=best_itr,
        #                                                                                             data_name='test')
        #print(val_accuracy_target, val_loss_target, best_cm_target)
        config_array = ["trail-" + str(trial_number), metric_name,

                        source_val_accuracy, source_val_loss, best_cm,

                        #val_accuracy_target, val_loss_target, best_cm_target,

                        best_classifier_loss, best_transfer_loss,
                        best_total_loss
                           ,
                        best_itr] + training_parameters
        config["trial_parameters_log"].writerow(config_array)

    if not is_training:
        # print("error")
        best_model = torch.load(model_path_for_testing)
        best_itr = config["best_itr"]

    post_training(best_model, best_itr, best_classifier_loss_numpy_acc, best_total_loss_numpy_acc, best_transfer_loss_numpy_acc, "Source Val_loss")
    # post_training(best_model_total,best_itr_total,best_classifier_loss_numpy_total,best_total_loss_numpy_total,best_transfer_loss_numpy_total,"Total")
    # post_training(best_model_transfer,best_itr_transfer,best_classifier_loss_numpy_transfer,best_total_loss_numpy_transfer,best_transfer_loss_numpy_transfer,"Transfer")

    return best_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset', type=str, default='COVID19', help="The dataset or source dataset used")
    parser.add_argument('--trail', type=str, default='mb', help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=0.005)

    args = parser.parse_args()

    seed = 0

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string = dt_string.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

    dataset = args.dset
    valid_or_test = ""  # "valid or "test" or "" if whole dataset
    test_on_source_ed3 = False
    is_training = True

    if valid_or_test == "test":
        is_training = False

    testing = "testing"
    # testing = "train"

    if testing == "testing":
        is_training = False

    print(dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    log_output_dir_root = '../logs/' + dataset + '/'
    results_output_dir_root = '../experimental results/' + dataset + '/'
    models_output_dir_root = '../models/' + dataset + '/'

    trial_number = args.trail + "_" + dataset + "_" + testing + "_" + dt_string

    if dataset == 'HCV':
        source_path = "../Data/HCV/txt_80_20_tile/HCV_train_80.txt"
        valid_source_path = "../Data/HCV/txt_80_20_tile/HCV_val_20.txt"
        target_path = "../Data/HCV/HCV_target_tile.txt"
        no_of_classes = 2

    elif dataset == 'HIV':
        source_path = "../Data/HIV/txt_80_20_tile/HIV_train_80.txt"
        valid_source_path = "../Data/HIV/txt_80_20_tile/HIV_val_20.txt"
        target_path = "../Data/HIV/HIV_target_tile.txt"
        no_of_classes = 2

    elif dataset == 'ZIKA':
        source_path = "../Data/ZIKA/txt_90_10_tile/ZIKA_train_90.txt"
        valid_source_path = "../Data/ZIKA/txt_90_10_tile/ZIKA_val_10.txt"
        target_path = "../Data/ZIKA/ZIKA_target_tile.txt"
        no_of_classes = 2

    elif dataset == 'HBV':
        source_path = "../Data/HBV/txt_80_20_tile/HBV_train_80.txt"
        valid_source_path = "../Data/HBV/txt_80_20_tile/HBV_val_20.txt"
        target_path = "../Data/HBV/HBV_target_tile.txt"
        no_of_classes = 2

    elif dataset == 'COVID19':
        source_path = "../Data/COVID19/txt_80_20_tile/COVID19_train_80.txt"
        valid_source_path = "../Data/COVID19/txt_80_20_tile/COVID19_val_20.txt"
        target_path = "../Data/COVID19/COVID19_target_tile.txt"
        no_of_classes = 2

    elif dataset == 'CAS12':
        source_path = "../Data/CAS12/txt_80_20_tile/CAS12_train_80.txt"
        valid_source_path = "../Data/CAS12/txt_80_20_tile/CAS12_val_20.txt"
        target_path = "../Data/CAS12/CAS12_target_tile.txt"
        no_of_classes = 2


    else:
        no_of_classes = None

    net = 'Xception'
    # net = 'ResNet50'
    dset = dataset

    lr_ = args.lr
    gamma = 0.001
    power = 0.75
    # power = 0.9

    momentum = 0.9
    weight_decay = 0.0005
    nesterov = True
    optimizer = optim.Adam

    config = {}
    config['method'] = 'CDAN+E'
    config["gpu"] = '0'
    config["num_iterations"] = 10000
    config["test_interval"] = 50
    config["snapshot_interval"] = 5000

    batch_size = 8
    batch_size_test = 128
    use_bottleneck = False
    bottleneck_dim = 256
    adv_lay_random = False
    random_dim = 1024
    new_cls = True

    if not is_training:
        valid_source_path = "../Data/Test/mb1/mb_test.txt"
        target_path = "../Data/Test/mb1/mb_test.txt"
        model_path_for_testing = "../Final Models/CDAN + GAN/COVID19/model_1600.pth.tar"

        config["num_iterations"] = 0
        best_itr = "testing"

        print("Testing:")
        config["best_itr"] = "testing"

    print("num_iterations", config["num_iterations"])

    header_list = ["trail no ", 'metric name',

                   'source_val_accuracy', 'source_val_loss', 'best_cm',

                   # 'val_accuracy_target', 'val_loss_target', 'best_cm_target',

                   "best_classifier_loss", "best_transfer_loss", "best_total_loss"
                      ,
                   "best_itr"] + \
                  ["lr", "gamma", "power", "momentum", "weight_decay", "nesterov", "optimizer",
                   "batch_size", "batch_size_test", "use_bottleneck", "bottleneck_dim", "adv_lay_random",
                   "random_dim",
                   "no_of_classes", "new_cls", "dset", "net", "source_path", "target_path", "output_path",
                   "model_path"
                      , "logs_path", "gpu", "test_interval", "seed"]

    log_output_path = log_output_dir_root + net + '/' + 'trial-' + trial_number + '/'
    trial_results_path = net + '/trial-' + trial_number + '/'
    config["output_path"] = results_output_dir_root + trial_results_path
    config["model_path"] = models_output_dir_root + trial_results_path
    config["logs_path"] = log_output_path
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])
    if is_training:
        if not os.path.exists(config["model_path"]):
            os.makedirs(config["model_path"])

        # if not os.path.exists(config["output_path"]):
        #     os.makedirs(config["output_path"])

    if not os.path.isfile(osp.join(log_output_dir_root, "log.csv")):
        with open(osp.join(log_output_dir_root, "log.csv"), mode='w') as param_log_file:
            param_log_writer = csv.writer(param_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            param_log_writer.writerow(header_list)

    config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")
    config["trial_parameters_log"] = csv.writer(open(osp.join(log_output_dir_root, "log.csv"), "a"))

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}

    if "Xception" in net:
        config["network"] = \
            {"name": network.XceptionFc,
             "params":
                 {"use_bottleneck": use_bottleneck,
                  "bottleneck_dim": bottleneck_dim,
                  "new_cls": new_cls}}
    elif "ResNet50" in net:
        config["network"] = {"name": network.ResNetFc,
                             "params":
                                 {"resnet_name": net,
                                  "use_bottleneck": use_bottleneck,
                                  "bottleneck_dim": bottleneck_dim,
                                  "new_cls": new_cls}}

    config["loss"]["random"] = adv_lay_random
    config["loss"]["random_dim"] = random_dim

    if optimizer == optim.SGD:

        config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': lr_, "momentum": momentum,
                                                                   "weight_decay": weight_decay, "nesterov": nesterov},
                               "lr_type": "inv",
                               "lr_param": {"lr": lr_, "gamma": gamma, "power": power}}

    elif optimizer == optim.Adam:
        config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': lr_,
                                                                    "weight_decay": weight_decay},
                               "lr_type": "inv",
                               "lr_param": {"lr": lr_, "gamma": gamma, "power": power}}

    config["dataset"] = dset
    config["data"] = {"source": {"list_path": source_path, "batch_size": batch_size},
                      "target": {"list_path": target_path, "batch_size": batch_size},
                      "test": {"list_path": target_path, "batch_size": batch_size_test},
                      "valid_source": {"list_path": valid_source_path, "batch_size": batch_size}}
    config["optimizer"]["lr_param"]["lr"] = lr_
    config["network"]["params"]["class_num"] = no_of_classes

    config["out_file"].write(str(config))
    config["out_file"].flush()

    training_parameters = [lr_, gamma, power, momentum, weight_decay, nesterov, optimizer,
                           batch_size, batch_size_test, use_bottleneck, bottleneck_dim, adv_lay_random, random_dim,
                           no_of_classes, new_cls, dset, net, source_path, target_path, config["output_path"],
                           config["model_path"]
        , config["logs_path"], config["gpu"], config["test_interval"], str(seed)]

    print("source_path", source_path)
    print("target_path", target_path)
    print("lr_", lr_)
    print('GPU', os.environ["CUDA_VISIBLE_DEVICES"], config["gpu"])

    train(config)
