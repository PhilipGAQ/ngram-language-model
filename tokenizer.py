# remove punctuations in the text
import os

def rmv_pct(train_path,path1,path2):

    with open(train_path, 'r',encoding='utf-8') as f:
        train = [l.strip() for l in f.readlines()]
    with open(path1, 'r',encoding='utf-8') as f:
        test1 = [l.strip() for l in f.readlines()]
    with open(path2,'r',encoding='utf-8') as f:
        test2=[l.strip() for l in f.readlines()]
    train = [i.replace('.','').replace(',','').replace('!','').replace('?','').replace('。','').replace('！','').replace('？','').replace('，','').replace('《','').replace('》','').replace('、','').replace('：','').replace('（','').replace('）','').replace('“','').replace('”','') for i in train]
    test1 = [i.replace('.','').replace(',','').replace('!','').replace('?','').replace('。','').replace('！','').replace('？','').replace('，','').replace('《','').replace('》','').replace('、','').replace('：','').replace('（','').replace('）','').replace('“','').replace('”','') for i in test1]
    test2 = [i.replace('.','').replace(',','').replace('!','').replace('?','').replace('。','').replace('！','').replace('？','').replace('，','').replace('《','').replace('》','').replace('、','').replace('：','').replace('（','').replace('）','').replace('“','').replace('”','') for i in test2]
    # create new files called data_new/train.txt
    if  not os.path.exists('data_new'):
        os.mkdir('data_new')

    with open('data_new/train.txt', 'w',encoding='utf-8') as f:
        for i in train:
            f.write(i+'\n')
    with open('data_new/test.1.txt', 'w',encoding='utf-8') as f:
        for i in test1:
            f.write(i+'\n')
    with open('data_new/test.2.txt', 'w',encoding='utf-8') as f:
        for i in test2:
            f.write(i+'\n')
    
    return

if __name__ == '__main__':
    train_path = os.path.join(os.getcwd(), 'data2/train.txt')
    test1_path = os.path.join(os.getcwd(), 'data2/test.1.txt')
    test2_path = os.path.join(os.getcwd(), 'data2/test.2.txt')
    rmv_pct(train_path,test1_path,test2_path)
    print("done!")

