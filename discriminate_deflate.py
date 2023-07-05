from transformers.models.t5 import T5ForConditionalGeneration, T5Config
from T5PegasusTokenizer import T5PegasusTokenizer
from collections import defaultdict
import datetime
import torch
import os
import jieba
from configuration_t5 import T5Config
import numpy as np
import random
from torch import nn
from rouge import Rouge
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

jieba.initialize()
model_path = 'fooT5'
config = T5Config.from_json_file(model_path + '/config.json')
model = T5ForConditionalGeneration(config)
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
original = torch.load('./models/t5_text_no_open_attribute_0906_60000.pt')
model.load_state_dict(original['state_dict']['network'])


def save(filename):
    params = {
        'state_dict': {
            'network': model.state_dict()
        },
        'epoch': 1
    }
    torch.save(params, filename)


def eval_a_line(dish,image_id,target):
    input_ids = tokenizer(dish, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(input_ids)

    res = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if res=='':
        return '无'
    else:
        return res


class Discriminate(nn.Module):

    def __init__(self,model_path):
        super(Discriminate, self).__init__()
        orignal = torch.load(model_path)
        model.load_state_dict(orignal['state_dict']['network'])
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        self.bert = model.encoder

    def forward(self, input_ids,attention_mask):
        feature = self.bert(input_ids = input_ids.cuda(),attention_mask = attention_mask.cuda())
        y = torch.mean(feature[0], dim=1)
        y_pred = self.mlp(y)
        return y_pred

    def predict(self,input_ids,attention_mask):
        pred = F.softmax(self.forward(input_ids,attention_mask),dim=1)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return ans

    def predict_1(self,input_ids,attention_mask):
        pred = self.forward(input_ids,attention_mask)
        ans = []
        for t in pred:
            if t.item()>0.50:
                ans.append(1)
            else:
                ans.append(0)
        return ans


with open('attribute_train.txt', 'r', encoding='utf-8') as f:
    attribute_data = eval(f.read())  # eval

attribute_dict = dict()
for i in attribute_data[0]:
    attribute_dict[i[0]] = set()


def train_dis(file_path,model_path):
    idx = 0
    today_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    import torch.optim as optim

    model.cuda()
    dis_model = Discriminate(model_path)

    dis_model.cuda()

    # loss_discriminate = nn.CrossEntropyLoss()
    loss_discriminate = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dis_model.mlp.parameters(), lr=0.0001)

    t5_app_d = './dis_data_0801'

    files = os.listdir(t5_app_d)

    save_text = []
    save_label = []

    best_res = 0
    sm = []
    for idx, ln in enumerate(open(file_path)):
        if idx == 0:
            continue
        info = ln.strip('\n').split("\t")

        sm.append(info)

    train_data = sm[:-1000]

    dataloader = DataLoader(dataset=sm[:-1000],
                            batch_size=16,
                            shuffle=False,
                            num_workers=0,
                            # sampler = train_sampler
                            )

    for epoch in range(5):
        index = 0

        dis_model.train()
        for ba in dataloader:

            text, img, label = ba
            img = torch.tensor([eval(g) for g in img])
            label = torch.tensor([int(l) for l in label])

            encoding = tokenizer(
                text,
                padding="longest",
                max_length=50,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )

            input_ids = encoding.input_ids
            attention_mask = encoding.attention_mask

            pre_y = dis_model(input_ids.cuda(), attention_mask.cuda())

            loss = loss_discriminate(pre_y.squeeze(1), label.float().cuda())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if index % 3 == 0:
                print(idx, loss.item())

                print('test_eval:          ')
                eval_res = test_dis(dis_model, sm[-1000:])

                if eval_res > best_res:
                    best_res = eval_res
                    print('best_res: ', best_res)
                    torch.save(dis_model.state_dict(), 'discriminate_BCE_0102.pth')

            index += 1


def test_dis(dis_model,test_dataset):

    dataloader = DataLoader(dataset=test_dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=0,
                            )
    target_y = []
    res = []
    for ba in dataloader:
        dis_model.eval()
        text, img, label = ba
        img = torch.tensor([eval(g) for g in img])
        label = torch.tensor([int(l) for l in label])

        encoding = tokenizer(
            text,
            padding="longest",
            max_length=50,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        pre_y = dis_model.predict_1(input_ids.cuda(),attention_mask.cuda())

        target_y.extend(label.tolist())
        res.extend(pre_y)

    print('f1:  ',f1_score(target_y, res))
    return f1_score(target_y, res)


def data_product():

    idx = 0
    today_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    import torch.optim as optim
    model.cuda()

    optimizer = optim.Adamax(model.parameters(), lr=0.0003)

    t5_app_d = './data/wai_csv_400wan_new_0831'

    files = os.listdir(t5_app_d)

    test_file = files[-1:]
    files = files[:5]

    sm = []
    for f in files[:1]:
        if len(sm) > 10000:
            break

        for idx, ln in tqdm(enumerate(open(t5_app_d + '/' + f))):
            if idx == 0:
                continue
            info = ln.strip('\n').split("\t")

            if isinstance(eval(info[-1]), dict):
                if len(info[0])<3:
                    continue
                if len(set(attribute_dict.keys())&set(eval(info[-1]).keys()))>6:
                    sm.append(info)

    dataloader = DataLoader(dataset=sm,
                            batch_size=64,
                            shuffle=False,
                            num_workers=0,
                            )

    key_name = ['text', 'picture', 'label']
    res = []
    for batch in tqdm(dataloader):

        name, _, _, _, _, _, _, _, _, _, pic_token, attribute = batch
        if pic_token == '':
            continue

        for batch_index in range(len(name)):

            if isinstance(eval(attribute[batch_index]), dict):
                attribute_have = eval(attribute[batch_index])
            else:
                attribute_have = dict()

            for attribute_key in attribute_dict.keys():
                save = []

                attribute_pre = eval_a_line('[unused8]' + name[batch_index] + '的' + attribute_key,
                                            pic_token[batch_index], '无')
                attribute_pre = ','.join(list(set(attribute_pre.replace(' ', '').split(','))))
                dis_text = '[unused8]' + name[batch_index] + '的' + attribute_key + '为' + attribute_pre
                save.append(dis_text)
                save.append(pic_token[batch_index])
                if attribute_key in attribute_have.keys():
                    save.append(1)
                elif attribute_key=='甜' and '蛋糕' in dis_text:
                    save.append(1)
                else:
                    save.append(0)

                res.append(save)

        test = pd.DataFrame(columns=key_name, data=res)
        test.to_csv('discriminate__train_100000.csv', sep='\t', index=0)


if __name__ == '__main__':

    data_product()

    train_dis('discriminate__train_100000.csv',)

