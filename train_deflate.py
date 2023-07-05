import pandas as pd
from T5PegasusTokenizer import T5PegasusTokenizer
from collections import defaultdict
import datetime
import torch
import os
import jieba
from configuration_t5 import T5Config
import numpy as np
import random
from rouge import Rouge
from model_t5 import T5ForConditionalGeneration, T5Model
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from eval_deflate import t5_eval
import torch.optim as optim

test_flag = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


jieba.initialize()

with open('attribute_train.txt', 'r', encoding='utf-8') as f:
    attribute_data = eval(f.read())  # eval

attribute_dict = dict()
for i in attribute_data[0]:
    attribute_dict[i[0]] = set()

model_path = 'fooT5'
config = T5Config.from_json_file(model_path + '/config.json')
model = T5ForConditionalGeneration(config)
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)


class SampleLine:
    def __init__(self, info):
        name, _, _, _, _, _, _, _, label_name, description, pic_token, attribute = info
        self.name = name

        self.label_name = label_name
        self.description = description
        if pic_token != '':
            self.pic_token = eval(pic_token)
        else:
            self.pic_token = ''
        if attribute != '':
            self.attribute = eval(attribute)
        else:
            self.attribute = attribute


class DishDescriptionDataset(Dataset):
    def __init__(self, fnm):
        self.smps = self.load_data(fnm)
        pass

    def load_data(self, fnm):
        smps = []
        for idx, ln in tqdm(enumerate(open(fnm))):
            if idx == 0:
                continue
            info = ln.strip('\n').split("\t")
            smp = SampleLine(info)
            smps.append(smp)
            if smp.pic_token=='':
                continue
            #
            if len(smps) > 100 and test_flag == 1:
                break
        return smps

    def __getitem__(self, index):
        return self.smps[index]

    def __len__(self):
        return len(self.smps)


class TaskGenerator:
    def __init__(self):
        self.attr_tp = self.load_attrs()

    def load_attrs(self):
        mp = {
            'taste': '口味',
            'scene': '场景',
            'ingredient': '食材',
            'cook_method': '做法'
        }

        attr_tp = defaultdict(list)
        for ln in open('data/distinct_attribute_tags.tsv'):
            attr, tp = ln.strip('\n').split('\t')
            tp = mp[tp]
            attr_tp[attr].append(tp)
        return attr_tp

    def match_attrs(self, text):
        len_text = len(text)
        s_idx = 0
        match_res = []
        while s_idx < len_text:
            cache = []
            span_length = 1
            while span_length < 4 and s_idx + span_length <= len_text:
                span = text[s_idx: s_idx + span_length]
                if span in self.attr_tp:
                    cache.append((span, self.attr_tp[span]))
                span_length += 1
            if len(cache) > 0:
                match_res.append(cache[-1])
                s_idx += len(cache[-1])
            else:
                s_idx += 1
        return match_res

    def basic_dish_info(self, smp, with_tag_name=False):
        source = ''
        target = smp.label_name

        if with_tag_name and random.random() > 0.5:
            source += (smp.label_name + ',')

        source += smp.name
        if len(smp.description)!=0:
            source+='('+smp.description+')'

        return source

    def task1(self, smp):
        task_token = '[unused7]'
        source = task_token + self.basic_dish_info(smp, with_tag_name=True)
        target = smp.description
        if len(target) == 0:
            assert 1 == 2, 'hi, no Input description'

        if random.random() < 0.6:
            m_res = self.match_attrs(target)
            if len(m_res):
                attr_span, tp_list = random.choice(m_res)
                tp = random.choice(tp_list)
                source += '[ununsed60]' + tp

        return source, target

    def task2(self, smp,k,v):
        task_token = '[unsed8]'
        source = task_token + self.basic_dish_info(smp, with_tag_name=True)
        source += '的'+k+'?'
        target = v
        return source, target

    def hub(self, smp):
        model_func = {
            1: self.task1,
            2: self.task2
        }

        source_list = []
        target_list = []
        label_list = []

        task_num = []
        if len(smp.description)!=0:
            aa,bb = model_func[1](smp)
            source_list.append(aa)
            target_list.append(bb)
            label_list.append(1)

        if isinstance(smp.attribute, dict):
            for k,v in smp.attribute.items():
                if k=='库存包装费设置方式' or k=='卡片展示字段' or k not in set(attribute_dict.keys()):
                    continue
                aa,bb = model_func[2](smp,k,v)
                source_list.append(aa)
                target_list.append(bb)
                label_list.append(1)

        return source_list,target_list,label_list


def tokenize(batch_smps):
    tg = TaskGenerator()
    max_source_length = 30 + 30
    max_target_length = 50
    des = []
    dishes = []
    pic_token = []
    neg = []
    neg_label = []
    neg_encoding = None

    attribute_label = []
    for smp in batch_smps:
        s, t ,l = tg.hub(smp)
        for pp in range(len(s)):
            pic_token.append(smp.pic_token)
        des.extend(t)
        dishes.extend(s)
        attribute_label.extend(l)

    encoding = tokenizer(
        dishes,
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    target_encoding = tokenizer(
        des, padding="longest", max_length=max_target_length, truncation=True
    )

    labels = target_encoding.input_ids
    labels = torch.tensor(labels)
    attribute_label = torch.tensor(attribute_label)
    neg_label = torch.tensor(neg_label)

    pic_token = torch.tensor(pic_token)

    return encoding, labels,pic_token,attribute_label,neg_encoding,neg_label


def train_mt5():
    idx = 0
    today_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    model.cuda()

    optimizer = optim.Adamax(model.parameters(), lr=0.0002)

    t5_app_d = './data/wai_csv_400wan_new_0831'


    files = os.listdir(t5_app_d)
    parts = len(files)


    test_file = files[-1:]
    files = files[:5]



    for epoch in range(5):

        print('epoch:',epoch)
        for idx1, fnm in enumerate(files):
            if fnm[-3:]!='csv':
                continue
            train_dataset = DishDescriptionDataset(t5_app_d + '/' + fnm)



            dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=4,
                                    collate_fn=tokenize,
                                    shuffle=False,
                                    num_workers=0,
                                    # sampler = train_sampler
                                    )
            for ba in tqdm(dataloader):
                model.train()
                encoding, labels, pic_token,attribute_label,neg_encoding, neg_label = ba
                input_ids = encoding.input_ids
                attention_mask = encoding.attention_mask

                loss = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(),
                             labels=labels[:, 1:].cuda(), img_ids=pic_token.cuda()).loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if idx % 2500 == 0:
                    print(idx, loss.item())
                    print('test_eval:          ')
                    t5_eval(model,tokenizer)
                    # save(f'models/mul/t5_mul_no_open_attribute_0114_multrain{idx}.pt')

                idx += 1


if __name__ == '__main__':
    train_mt5()



