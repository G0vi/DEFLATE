import torch
import  numpy as np
import pandas
from torch import nn
from configuration_t5 import T5Config
from model_t5 import T5ForConditionalGeneration, T5Model
import torch.nn.functional as F

from T5PegasusTokenizer import T5PegasusTokenizer



class Discriminate(nn.Module):

    def __init__(self):
        super(Discriminate, self).__init__()
        model_path = 'fooT5'
        # model = MT5ForConditionalGeneration.from_pretrained(model_path)
        config = T5Config.from_json_file(model_path + '/config.json')
        g_model = T5ForConditionalGeneration(config)
        parameter = torch.load('./models/save/t5_mul_no_open_attribute_0903_120000.pt')
        g_model.load_state_dict(parameter['state_dict']['network'])
        self.mlp = nn.Sequential(
            # nn.Linear(2, 4),  # PyTorch 中的线性层，wx + b
            # nn.Tanh(),
            nn.Linear(config.d_model, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        # self.linear = torch.nn.Linear(768, )
        # self.sigmoid = torch.nn.Sigmoid()
        self.bert = g_model.encoder

    def forward(self, input_ids,attention_mask,img_ids):
        feature = self.bert(input_ids = input_ids.cuda(),attention_mask = attention_mask.cuda(),img_ids = img_ids)
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

    def predict_1(self,input_ids,attention_mask,img_ids):
        pred = self.forward(input_ids,attention_mask,img_ids)
        ans = []
        for t in pred:
            if t.item()>0.50:
                ans.append(1)
            else:
                ans.append(0)
        return ans


def get_f1(pred_num, gold_num, right_num, right_num2=None):
    p = right_num / pred_num if pred_num else 0
    if right_num2:
        right_num = right_num2
    r = right_num / gold_num if gold_num else 0
    f1 = 2 * p * r / (p + r) if p * r else 0
    return p * 100, r * 100, f1 * 100

def get_attr_score(pred, gold, return_nums=False):
    gold_attr_set = set(list(gold.keys()))
    pred_attr_set = set(list(pred.keys()))
    join = gold_attr_set & pred_attr_set
    if return_nums:
        return len(pred_attr_set), len(gold_attr_set), len(join)
    score1 = get_f1(len(pred_attr_set), len(gold_attr_set), len(join))
    return score1

def is_fuzzy_match(first_string, second_string):
    # 判断两个字符串是否模糊匹配;标准是最长公共子串覆盖第一个标签的长度至少一半
    first_str = first_string.strip()
    second_str = second_string.strip()

    len_1 = len(first_str)
    len_2 = len(second_str)
    len_vv = []
    global_max = 0
    for i in range(len_1 + 1):
        len_vv.append([0] * (len_2 + 1))
    if len_1 == 0 or len_2 == 0:
        return False
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if first_str[i - 1] == second_str[j - 1]:
                len_vv[i][j] = 1 + len_vv[i - 1][j - 1]
            else:
                len_vv[i][j] = 0
            global_max = max(global_max, len_vv[i][j])
    # if global_max >= 2 and global_max / len(first_str) >= 0.5:
    if global_max / len(first_str) > 0.5:
        return True
    return False

def get_one_instance_partial_score(pred, gold, return_nums=False):
    gold_num = 0
    pred_num = 0
    right_num1 = 0
    right_num2 = 0

    for attr in pred:
        cur_value = set(pred[attr])
        pred_num += len(cur_value)
        if attr not in gold:
            continue
        for eve_pred in cur_value:
            flag = 0
            for eve_gold in gold[attr]:
                if is_fuzzy_match(eve_gold, eve_pred):
                    flag = 1
                    break
            if flag:
                right_num1 += 1

    for attr in gold:
        cur_value = set(gold[attr])
        gold_num += len(cur_value)
        if attr not in pred:
            continue
        for eve_gold in cur_value:
            flag = 0
            for eve_pred in pred[attr]:
                if is_fuzzy_match(eve_gold, eve_pred):
                    flag = 1
                    break
            if flag:
                right_num2 += 1
    if return_nums:
        return pred_num, gold_num, right_num1, right_num2
    # print(pred_num, gold_num, right_num1, right_num2)
    score = get_f1(pred_num, gold_num, right_num1, right_num2=right_num2)
    return score


# 获取某条样本的完美匹配的score或者num
def get_one_instance_score(pred: dict, gold: dict, return_nums=False):
    gold_num = 0
    pred_num = 0
    right_num = 0
    gold_set = {}
    for attr in gold:
        cur_value = set(gold[attr])
        # print('cur_value is', gold, cur_value)
        gold_num += len(cur_value)
        gold_set[attr] = cur_value
    for attr in pred:
        cur_value = set(pred[attr])
        pred_num += len(cur_value)
        if attr not in gold_set:
            continue
        for eve in cur_value:
            if eve in gold_set[attr]:
                right_num += 1
    if return_nums:
        return pred_num, gold_num, right_num
    score = get_f1(pred_num, gold_num, right_num)
    return score





def score_jd(target,predict,mode='attr'):
    f1 = 0
    p,r = 0,0
    pred, gold, right = 0,0,0
    right2=0
    for i in range(len(target)):

        if mode=='attr':
            y_true = target[i]
            y_predict = predict[i]
            pred_num, gold_num, right_num = get_attr_score(y_predict, y_true, return_nums=True)

        elif mode=='partial':
            y_true = target[i]
            y_predict = predict[i]
            pred_num, gold_num, right_num,right_num2 = get_one_instance_partial_score(y_predict, y_true, return_nums=True)

        else:
            y_true = target[i]
            y_predict = predict[i]
            pred_num, gold_num, right_num = get_one_instance_score(y_predict, y_true, return_nums=True)

        pred += pred_num
        gold += gold_num
        right += right_num
        if mode=='partial':
            right2+=right_num2


    if mode == 'partial':
        p, r, f1 = get_f1(pred, gold, right, right_num2=right2)
    else:
        p, r, f1 = get_f1(pred, gold, right, right_num2=None)
    # print('f1：        ',f1)
    return f1,p,r


def eval_a_line(dish, image_id, target,tokenizer,model):
    input_ids = tokenizer(dish, return_tensors="pt").input_ids.cuda()

    outputs = model.generate(input_ids, img_ids=image_id)

    # print(input_ids, outputs)

    res = tokenizer.decode(outputs[0], skip_special_tokens=True)


    if res == '':

        return '无'

    else:

        return res

def t5_eval(model,tokenizer):
    path = 'eval_data.csv'
    dis_model = Discriminate()
    dis_model_parameter = torch.load('discriminate_BCE_1115_mul.pth')
    dis_model.load_state_dict(dis_model_parameter)
    dis_model.cuda()
    pic_data_dict = np.load('pic_data.npy', allow_pickle=True).item()
    target_text = []
    predict_text = []

    model.eval()
    model.cuda()

    pd_reader = pandas.read_csv(path)
    name_list = pd_reader.context
    img_list = pd_reader.url
    attr_list = pd_reader.attribute_output

    res = []
    target_y = []
    predict_y = []
    name_save = []

    target_text_product = []
    predict_text_product = []
    target_y_product = []
    predict_y_product = []
    predict_product = dict()
    target_product = dict()

    for i in range(len(pd_reader.context)):
        if i % 1000 == 0:
            print('nums:      ', i)
        # if i>9:
        #     break

        name = name_list[i]

        if name not in name_save:
            name_save.append(name)
            if len(list(target_product.keys())) != 0:
                target_text.append(target_product)
                predict_text.append(predict_product)

            predict_product = dict()
            target_product = dict()

        img = img_list[i]
        predict_att, predict_res = attr_list[i].split(":")
        predict_res = predict_res.strip(',')

        pic_data = pic_data_dict[img]
        pic_data = torch.tensor(pic_data).unsqueeze(0).cuda()

        attribute_pre = eval_a_line('[unused8]' + name + '的' + predict_att,
                                    pic_data, predict_res,tokenizer,model)
        attribute_pre = ','.join(list(set(attribute_pre.replace(' ', '').split(','))))

        predict_have = '[unused8]' + name + '的' + predict_att + '为' + attribute_pre

        encoding = tokenizer(
            predict_have,
            padding="longest",
            max_length=50,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        pre_y = dis_model.predict_1(input_ids.cuda(), attention_mask.cuda(), pic_data)

        if pre_y[0] == 0:
            attribute_pre = '无'


        res.append([name, predict_att, predict_res, attribute_pre])

        if attribute_pre != '无':
            # predict_y_product.append(predict_att)
            # predict_text_product.extend(attribute_pre.replace(' ','').split(','))
            predict_product[predict_att] = attribute_pre.replace(' ', '').split(',')

        if predict_res != '无':
            # target_y_product.append(predict_att)
            # target_text_product.extend(predict_res.replace(' ','').split(','))
            target_product[predict_att] = predict_res.replace(' ', '').split(',')

    target_text.append(target_product)
    predict_text.append(predict_product)


    f1, p, r = score_jd(target_text, predict_text, 'value')
    print('value f1:  ', f1, p, r)
    f1, p, r = score_jd(target_text, predict_text, 'partial')
    print('partial f1:  ', f1, p, r)
    f1, p, r = score_jd(target_text, predict_text)
    print('attr f1:  ', f1, p, r)

    return

if __name__ == '__main__':


    model_path = 'fooT5'
    config = T5Config.from_json_file(model_path + '/config.json')
    eval_model = T5ForConditionalGeneration(config)
    tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
    model_data = torch.load('./models/save/t5_mul_no_open_attribute_0903_120000.pt')
    eval_model.load_state_dict(model_data['state_dict']['network'])
    t5_eval(eval_model,tokenizer)