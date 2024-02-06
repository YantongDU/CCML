# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.MetaUtils
##########################
"""
import importlib
from collections import OrderedDict
import os,pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.config import Config
from recbole.data.dataloader import *
from recbole.utils.argument_list import dataset_arguments
from recbole.utils import set_color
from recbole.utils import init_logger, init_seed
import random

class Task():
    '''
    Task is the basis of meta learning.
    For user cold start recsys, a task usually refers to a user.
    '''
    def __init__(self,taskInfo,spt,qrt):
        '''
        Generate a new task, including task information, support set, query set.

        :param taskInfo(dict): eg. {'task_id':202} task_id always equals to user_id
        :param spt(Interaction): eg. (user_id, item_id, rating) tuples
        :param qrt(Interaction): For training eg. (user_id, item_id, rating) tuples; for testing (user_id, item_id) tuples
        '''
        self.taskInfo = taskInfo
        self.spt = spt
        self.qrt = qrt

def create_meta_dataset(config):
    '''
    This function is rewritten from 'recbole.data.create_meta_dataset(config)'
    '''
    multi = config['multi']
    if not multi:
        from MetaDataset import MetaDataset
    else:
        from MultiMetaDataset2 import MetaDataset

    dataset_class =MetaDataset

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset

def meta_data_preparation(config, dataset):
    '''
    This function is rewritten from 'recbole.data.data_preparation(config, dataset)'
    '''
    from recbole.data.utils import load_split_dataloaders, save_split_dataloaders
    from MetaDataLoader import MetaDataLoader

    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        # print(train_dataset.user_num)  There some problems with incorrect user number in .inter sets.

        train_sampler, valid_sampler, test_sampler = None,None,None

        train_data = MetaDataLoader(config, train_dataset, train_sampler, shuffle=True)
        valid_data = MetaDataLoader(config, valid_dataset, valid_sampler, shuffle=True)
        test_data = MetaDataLoader(config, test_dataset, test_sampler, shuffle=True)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data

def meta_data_preparation_multi(config, dataset, dataset_nc, dataset_uc, dataset_ic, dataset_uic):
    '''
    This function is rewritten from 'recbole.data.data_preparation(config, dataset)'
    '''
    from recbole.data.utils import load_split_dataloaders, save_split_dataloaders
    from recbole_ccml.MetaDataLoader import MetaDataLoader

    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        built_datasets = dataset.build()
        nc_test_datasets = dataset_nc.build(state_type='testing')
        uc_test_datasets = dataset_uc.build(state_type='testing')
        ic_test_datasets = dataset_ic.build(state_type='testing')
        uic_test_datasets = dataset_uic.build(state_type='testing')

        train_dataset, valid_dataset, _ = built_datasets
        nc_test_dataset, _, _ = nc_test_datasets
        uc_test_dataset, _, _ = uc_test_datasets
        ic_test_dataset, _, _ = ic_test_datasets
        uic_test_dataset, _, _ = uic_test_datasets
        # print(train_dataset.user_num)  There some problems with incorrect user number in .inter sets.

        train_sampler, valid_sampler, test_sampler = None,None,None

        train_data = MetaDataLoader(config, train_dataset, train_sampler, shuffle=True, training=True)
        valid_data = MetaDataLoader(config, valid_dataset, valid_sampler, shuffle=True)
        nc_test_data = MetaDataLoader(config, nc_test_dataset, test_sampler, shuffle=True)
        uc_test_data = MetaDataLoader(config, uc_test_dataset, test_sampler, shuffle=True)
        ic_test_data = MetaDataLoader(config, ic_test_dataset, test_sampler, shuffle=True)
        uic_test_data = MetaDataLoader(config, uic_test_dataset, test_sampler, shuffle=True)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, nc_test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, nc_test_data, uc_test_data, ic_test_data, uic_test_data

class GradCollector():
    '''
    This is a common data struct to collect grad.

    For the sake of complex calculation graph in meta learning, we construct this data struct to
    do grad operations on batch data.
    '''
    def __init__(self,paramsNameList):
        '''
        Initialize GradCollector Object.
        :param paramsNameList: Usually comes from list(nn.Moudule.state_dict().keys())
        '''
        self.paramNameList=paramsNameList
        self.gradDict=OrderedDict()
        self.taskGradDict=OrderedDict()

    def addGrad(self, gradTuple):
        '''
        Add grad and exist grad.

        :param gradTuple(tuple of torch.Tensor): Usually refers to grad tuple from 'torch.autograd.grad()'.

        '''
        for index,name in enumerate(self.paramNameList):
            if name not in self.gradDict:
                self.gradDict[name] = gradTuple[index]
            else:
                self.gradDict[name] += gradTuple[index]

    def addGradWithWeigth(self, weight):
        for taskId, gradTuple in self.taskGradDict.items():
            for index,name in enumerate(self.paramNameList):
                if name not in self.gradDict:
                    self.gradDict[name] = gradTuple[index] * weight[taskId]
                else:
                    self.gradDict[name] += gradTuple[index] * weight[taskId]

    def addAndSaveTaskGrad(self, taskIndex, grad):
        '''
        暂存task的梯度，用来计算task相似度
        Args:
            taskIndex:task编号
            grad: taskIndex对应task的梯度
        '''
        self.taskGradDict[taskIndex] = grad
        self.addGrad(grad)

    def averageGrad(self,size):
        '''
        Average operation for all grads.

        :param size: The denominator of average.

        '''
        for name,value in self.gradDict.items():
            self.gradDict[name]=self.gradDict[name]/size
        return self.gradDict


    def clearGrad(self):
        '''
        Clear all grads.
        '''
        self.gradDict = OrderedDict()

    def dumpGrad(self, clearTaskGrad=False):
        '''
        Return the grad tuple in the collector and clear all grads.

        :return grad(tuple of torch.Tensor): The grad tuple in the collector.

        '''
        grad=self.gradDict
        self.clearGrad()
        if clearTaskGrad:
            self.taskGradDict = OrderedDict()
        return grad

    def getTaskGradDict(self):
        return self.taskGradDict

    def print(self):
        '''
        Print name and grad.shape for all parameters.
        '''
        for name,grad in self.gradDict.items():
            print(name,grad.shape)

class EmbeddingTable(nn.Module):
    '''
    This is a data struct to embedding interactions.
    It supports 'token' and 'float' type.
    '''
    def __init__(self, embeddingSize, dataset, source=None):
        super(EmbeddingTable, self).__init__()

        if source is None:
            source = [FeatureSource.USER, FeatureSource.ITEM]
        self.dataset=dataset
        self.embeddingSize=embeddingSize

        self.embeddingDict = dict()
        self.initialize(source)

    def initialize(self,source):
        '''
        Initialize the fields of embedding.
        '''
        self.embeddingFields = self.dataset.fields(source=source)
        for field in self.embeddingFields:
            field_num = self.dataset.num(field) if self.dataset.field_num_dict is None else self.dataset.field_num_dict[field]
            if self.fieldType(field) is FeatureType.TOKEN:
                self.embeddingDict[field]=nn.Embedding(field_num,self.embeddingSize)
                self.add_module(field,self.embeddingDict[field])
            if self.fieldType(field) is FeatureType.TOKEN_SEQ:
                self.embeddingDict[field] = nn.Embedding(field_num, self.embeddingSize)
                self.add_module(field, self.embeddingDict[field])

    def fieldType(self,field):
        '''
        Convert field to type.
        :param field(str): Field name.
        :return type(str): Field type.
        '''
        return self.dataset.field2type[field]

    def getAllDim(self):
        dim=0
        for field in self.embeddingFields:
            if self.fieldType(field) is FeatureType.TOKEN or self.fieldType(field) is FeatureType.TOKEN_SEQ:
                dim+=self.embeddingSize
            if self.fieldType(field) is FeatureType.FLOAT:
                dim+=1
        return dim

    def getUserEmdDim(self):
        dim=0
        embFields = self.dataset.fields(source=[FeatureSource.USER])
        for field in embFields:
            if self.fieldType(field) is FeatureType.TOKEN or self.fieldType(field) is FeatureType.TOKEN_SEQ:
                dim+=self.embeddingSize
            if self.fieldType(field) is FeatureType.FLOAT:
                dim+=1
        return dim

    def getItemEmdDim(self):
        dim=0
        embFields = self.dataset.fields(source=[FeatureSource.ITEM])
        for field in embFields:
            if self.fieldType(field) is FeatureType.TOKEN or self.fieldType(field) is FeatureType.TOKEN_SEQ:
                dim+=self.embeddingSize
            if self.fieldType(field) is FeatureType.FLOAT:
                dim+=1
        return dim

    def getEmdWeight(self, fields):
        emb=[]
        embFields = fields
        for field in embFields:
                emb.append(self.embeddingDict[field].weight)
        f_emb = torch.cat(emb)
        return f_emb


    def embeddingSingleField(self, field, batchX, paramWeightDict=None):
        '''
        Embedding a single field.
        If the field type is 'float' then return itself, else return the 'token' embedding vectors.

        :param field(str): Field name.
        :param batchX(torch.Tensor):  Batch of tensor.
        :return: batchX(torch.Tensor): Batch of tensor.
        '''
        if self.fieldType(field) is FeatureType.TOKEN:
            # return self.embeddingDict[field](batchX) if paramWeightDict is None and paramWeightDict[field] is not None else F.embedding(batchX, paramWeightDict[field])
            return F.embedding(batchX, paramWeightDict[field]) if paramWeightDict is not None and field in paramWeightDict else self.embeddingDict[field](batchX)
        if self.fieldType(field) is FeatureType.TOKEN_SEQ:
            return torch.sum(F.embedding(batchX, paramWeightDict[field]) if paramWeightDict is not None and field in paramWeightDict else self.embeddingDict[field](batchX),
                             dim=1)
        if self.fieldType(field) is FeatureType.FLOAT:
            return torch.reshape(batchX, shape=(batchX.shape[0], 1))

    def embeddingAllFields(self,interaction, paramWeightDict=None):
        '''
        Embedding all fields of the interaction.
        Only fields in 'self.embeddingFields' will be embedded.

        :param interaction(Interaction): The input interaction.
        :return batchX(torch.Tensor): The concatenating process embedding of all fields.
        '''
        batchX=[]
        for field in self.embeddingFields:
            feature=self.embeddingSingleField(field,interaction[field], paramWeightDict=paramWeightDict)
            batchX.append(feature)
        batchX=torch.cat(batchX,dim=1)
        return batchX

    def forward(self,interaction):
        return self.embeddingAllFields(interaction)

class MetaParams():
    def __init__(self,paramStateDict):
        self.nameList=[]
        self.paramValue=[]
        self.paramShape=OrderedDict()
        for name,value in paramStateDict.items():
            self.nameList.append(name)
            self.paramValue.append(torch.reshape(value,(-1,)))
            self.paramShape[name]=value.shape
        self.paramValue=torch.cat(self.paramValue)

    def update(self,paramStateDict):
        self.nameList = []
        self.paramValue = []
        self.paramShape = OrderedDict()
        for name, value in paramStateDict.items():
            self.nameList.append(name)
            self.paramValue.append(torch.reshape(value, (-1,)))
            self.paramShape[name] = value.shape
        self.paramValue = torch.cat(self.paramValue)

    def destruct(self,paramValue=None):
        if paramValue is None:
            paramValue=self.paramValue
        params=OrderedDict()
        base=0
        for name in self.nameList:
            vol=torch.prod(torch.tensor(self.paramShape[name]))
            value=paramValue[base:base+vol]
            base+=vol
            params[name]=value.reshape(self.paramShape[name])

        return params

    def destructGrad(self,grad):
        output=[]
        base=0
        for name in self.nameList:
            vol = torch.prod(torch.tensor(self.paramShape[name]))
            value=grad[base:base+vol]
            base+=vol
            output.append(value.reshape(self.paramShape[name]))
        return tuple(output)

    def getGradVector(self,f,model,retain=False):
        grad = torch.autograd.grad(f,model.parameters(),create_graph=retain,retain_graph=retain)
        gradVector =torch.cat([torch.reshape(g,(-1,)) for g in grad])
        return gradVector

    def getOneStepSgdOutput(self,loss,model,lr,retain=False):
        if len(loss.shape) == 0:
            return self.destruct(self.paramValue-lr*self.getGradVector(loss,model,retain))

    def getHessianMatrix(self,loss,model):
        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
        grad = torch.cat([torch.reshape(g, (-1,)) for g in grad])
        hessianMatrix = []
        for item in grad:
            secondGrad = torch.autograd.grad(item, model.parameters(), retain_graph=True, create_graph=True)
            secondGrad = torch.cat([torch.reshape(g, (-1,)) for g in secondGrad])
            hessianMatrix.append(secondGrad)
        hessianMatrix = torch.stack(hessianMatrix)
        return hessianMatrix

    def getTaskGrad(self,loss_spt,model_spt,loss_qrt,model_qrt):
        part1=self.getGradVector(loss_qrt,model_qrt)
        hessianMatrix=self.getHessianMatrix(loss_spt,model_spt)
        part2=torch.eye(hessianMatrix.shape[0])-hessianMatrix
        taskGrad=torch.matmul(part2,part1).detach()
        return self.destructGrad(taskGrad)

    def get(self):
        return self.destruct()

    def print(self):
        params=self.destruct()
        for name,value in params.items():
            print(name,value)

def metaQuickStart(modelName,datasetName, paramConfig):
    if datasetName != 'book-crossing' and datasetName != 'book-crossing-CTR':
        configPath = ['recbole_ccml/model/' + modelName + '/' + modelName + '.yaml']
    else:
        configPath = ['recbole_ccml/model/' + modelName + '/' + modelName + '-BK.yaml']

    trainerClass = importlib.import_module('recbole_ccml.model.' + modelName + '.' + modelName + 'Trainer').__getattribute__(
        modelName + 'Trainer')
    modelClass = importlib.import_module('recbole_ccml.model.' + modelName + '.' + modelName).__getattribute__(modelName)

    paramConfig['data_path'] = 'recbole_ccml/dataset/'
    config = Config(model=modelClass, dataset=datasetName, config_file_list=configPath, config_dict=paramConfig)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    from recbole_ccml.MultiMetaDataset2 import MetaDataset

    dataset = MetaDataset(config)
    dataset_nc = MetaDataset(config, state_type='non_cold_testing')
    dataset_uc = MetaDataset(config, state_type='user_cold_testing')
    dataset_ic = MetaDataset(config, state_type='item_cold_testing')
    dataset_uic = MetaDataset(config, state_type='user_and_item_cold_testing')

    dataset.count_field_max_num((dataset, dataset_nc, dataset_uc, dataset_ic, dataset_uic))

    train_data, valid_data, nc_test_data, uc_test_data, ic_test_data, uic_test_data = meta_data_preparation_multi(config, dataset, dataset_nc, dataset_uc, dataset_ic, dataset_uic)

    logger.info(dataset)
    logger.info(train_data)

    # model loading and initialization
    model = modelClass(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = trainerClass(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data,show_progress=True)

    # model evaluation
    _, checkpoint = trainer.load_best_checkpoint()
    nc_test_result = trainer.evaluate(nc_test_data, checkpoint=checkpoint)
    uc_test_result = trainer.evaluate(uc_test_data, checkpoint=checkpoint)
    ic_test_result = trainer.evaluate(ic_test_data, checkpoint=checkpoint)
    uic_test_result = trainer.evaluate(uic_test_data, checkpoint=checkpoint)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('non_cold_test result: {}'.format(nc_test_result))
    logger.info('user_cold_test result: {}'.format(uc_test_result))
    logger.info('item_cold_test result: {}'.format(ic_test_result))
    logger.info('user_and_item_cold_test result: {}'.format(uic_test_result))

    return config, (nc_test_result, uc_test_result, ic_test_result, uic_test_result)
