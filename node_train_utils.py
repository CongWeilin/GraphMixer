from model import NodeClassificationModel
from construct_subgraph import NegLinkSampler
from sklearn.metrics import average_precision_score, f1_score
import copy

import torch
import pickle 
import os

@torch.no_grad()
def evaluate(model, all_node_embeds, all_labels, args):
    accs = list()
    for node_embeds, label in zip(all_node_embeds, all_labels):
        node_embeds, label = node_embeds.cuda(), label.cuda()
        pred = model(node_embeds)
        if args.posneg:
            acc = average_precision_score(label.cpu(), pred.softmax(dim=1)[:, 1].cpu())
        else:
            acc = f1_score(label.cpu(), torch.argmax(pred, dim=1).cpu(), average="micro")
            
        accs.append(acc)
    acc = float(torch.tensor(accs).mean())
    return acc

def fetch_eval_data(args, minibatch, neg_node_sampler, 
                    node_embeds_neg, node_labels_neg, over_sample=1): 
    # lets over sample some negative nodes to make the prediction more stable. 
    # otherwise, its very sensitive to what negative nodes are sampled
    
    fn = 'DATA/%s/node_cls_over_sample_%d'%(args.data, over_sample)
    if args.posneg:
        fn += '_posneg.pickle'
    else:
        fn += '.pickle'
        
    if os.path.exists(fn):
        all_data = pickle.load(open(fn, 'rb'))
    else:
        # valid nodes 
        valid_node_embeds = []
        valid_labels = []
        
        # test nodes
        test_node_embeds = []
        test_labels = []

        for _ in range(over_sample):
            # valid nodes 
            minibatch.set_mode('val')
            for node_embeds, label in minibatch:
                if args.posneg:
                    neg_idx = neg_node_sampler.sample(node_embeds.shape[0])
                    node_embeds = torch.cat([node_embeds, node_embeds_neg[neg_idx]], dim=0)
                    label = torch.cat([label, node_labels_neg[neg_idx]], dim=0)
                valid_node_embeds.append(node_embeds.cpu())
                valid_labels.append(label.cpu())
            # test nodes
            minibatch.set_mode('test')
            for node_embeds, label in minibatch:
                if args.posneg:
                    neg_idx = neg_node_sampler.sample(node_embeds.shape[0])
                    node_embeds = torch.cat([node_embeds, node_embeds_neg[neg_idx]], dim=0)
                    label = torch.cat([label, node_labels_neg[neg_idx]], dim=0)
                test_node_embeds.append(node_embeds.cpu())
                test_labels.append(label.cpu())
        all_data = valid_node_embeds, valid_labels, test_node_embeds, test_labels
        pickle.dump(all_data, open(fn, 'wb'))
    return all_data
        
        
    
    
def node_classification(args, node_embeds, node_role, node_labels):
    
    # create node classification model
    model = NodeClassificationModel(node_embeds.shape[1], 
                                    100, 
                                    node_labels.max() + 1).cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.posneg: # args.posneg makes sure the number of positive nodes == negative nodes, which is important for REDDIT + WIKI because its label is extremely unbalanced.
        node_role = node_role[node_labels == 1] # select all positive nodes
        node_embeds_neg = node_embeds[node_labels == 0].cuda() # select all negative node's embeddings
        node_embeds = node_embeds[node_labels == 1] # select all positive node's embeddings
        node_labels = torch.ones(node_embeds.shape[0], dtype=torch.int64).cuda()
        node_labels_neg = torch.zeros(node_embeds_neg.shape[0], dtype=torch.int64).cuda()
        neg_node_sampler = NegLinkSampler(node_embeds_neg.shape[0])

    # Setup mini-batch
    minibatch = NodeEmbMinibatch(node_embeds, node_role, node_labels, args.batch_size) # sample positive embeddings
    valid_node_embeds, valid_labels, test_node_embeds, test_labels = fetch_eval_data(args, minibatch, neg_node_sampler, node_embeds_neg, node_labels_neg, over_sample=1)
        
    best_epoch = 0
    best_acc = 0
    
    epoch = 0
    while True:
        epoch += 1
        ##########################################################
        minibatch.set_mode('train')
        minibatch.shuffle()
        model.train()
    
        optimizer.zero_grad() # try to use a very large batch-size to see if the result could get stable.
        for node_embeds, label in minibatch:

            if args.posneg:
                neg_idx = neg_node_sampler.sample(node_embeds.shape[0]) # sample a set of negative nodes with size equals to the positive node size
                node_embeds = torch.cat([node_embeds, node_embeds_neg[neg_idx]], dim=0)
                label = torch.cat([label, node_labels_neg[neg_idx]], dim=0)

            # forward + backward
            pred = model(node_embeds)
            loss = loss_fn(pred, label.long())
            loss.backward()
        optimizer.step()
        ##########################################################
        model.eval()
        valid_acc = evaluate(copy.deepcopy(model), valid_node_embeds, valid_labels, args)
        
        if epoch % 20 == 0:
            print('Epoch: {}\tVal acc: {:.4f}'.format(epoch, valid_acc))
            
        if valid_acc > best_acc:
            best_epoch = epoch
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

        if epoch - 500 > best_epoch:
            print('best_epoch', best_epoch)
            break

    print('Loading model at epoch {}...'.format(best_epoch))

    minibatch.set_mode('test')
    best_model.eval()
    test_acc = evaluate(best_model, test_node_embeds, test_labels, args)
    print('Testing acc: {:.4f}'.format(test_acc))

    
    
class NodeEmbMinibatch():

    def __init__(self, node_embeds, node_role, label, batch_size):
        self.node_role = node_role
        self.label = label
        self.batch_size = batch_size
        
        self.train_node_embeds = node_embeds[node_role == 0]
        self.val_node_embeds = node_embeds[node_role == 1]
        self.test_node_embeds = node_embeds[node_role == 2]
        
        self.train_label = label[node_role == 0]
        self.val_label = label[node_role == 1]
        self.test_label = label[node_role == 2]
        
        self.mode = 0
        self.s_idx = 0

    def shuffle(self):
        perm = torch.randperm(self.train_node_embeds.shape[0])
        self.train_node_embeds = self.train_node_embeds[perm]
        self.train_label = self.train_label[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
        elif mode == 'val':
            self.mode = 1
        elif mode == 'test':
            self.mode = 2
        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            node_embeds = self.train_node_embeds
            label = self.train_label
        elif self.mode == 1:
            node_embeds = self.val_node_embeds
            label = self.val_label
        else:
            node_embeds = self.test_node_embeds
            label = self.test_label
        if self.s_idx >= node_embeds.shape[0]:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, node_embeds.shape[0])
            curr_node_embeds = node_embeds[self.s_idx:end]
            curr_label = label[self.s_idx:end]
            self.s_idx += self.batch_size
            return curr_node_embeds.cuda(), curr_label.cuda()