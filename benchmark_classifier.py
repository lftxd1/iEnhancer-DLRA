from random import shuffle
from turtle import shape
from numpy import dtype
import tensorflow as tf
from sklearn.model_selection import KFold
import sys
sys.path.append('.')
from utils.MyUtils import *


def readTxt(file):
    with open(file,'r',encoding="utf-8") as f:
        return f.read().split("\n")

dic={'A':'A','T':'T','G':'G','C':'C'}
dic_con={'A':1,'T':2,'G':3,'C':4}


def readTxt(file):
    with open(file,'r',encoding="utf-8") as f:
        return f.read().split("\n")
non_test=readTxt("datasets/non_test.txt")
non_train=readTxt("datasets/non_train.txt")

strong_test=readTxt("datasets/strong_test.txt")
strong_train=readTxt("datasets/strong_train.txt")
weak_test=readTxt("datasets/weak_test.txt")
weak_train=readTxt("datasets/weak_train.txt")

def GenerateFromTextToNumpy(label,train):
    train_con=[]
    train_text=[]
    train_y=[]
    train_text_5=[]
    for i in train:
        t=threeSequecne(i,4) 
        train_text.append(np.array(t))

        t=threeSequecne(i,7) 
        train_text_5.append(np.array(t))

        con_t=[dic_con[key] for key in i]
        train_con.append(np.array(con_t))
        
        train_y.append(np.array([label]))
    train_con=np.array(train_con)
    train_text=np.array(train_text)
    train_y=np.array(train_y)
    train_text_5=np.array(train_text_5)
    return (train_con,train_text,train_y,train_text_5)


non_train_data=GenerateFromTextToNumpy(0,non_train)
strong_train_data=GenerateFromTextToNumpy(1,strong_train)
weak_train_data=GenerateFromTextToNumpy(0,weak_train)
feature_text=np.concatenate((strong_train_data[1],weak_train_data[1]))
train_con=np.concatenate((strong_train_data[0],weak_train_data[0]))[:,:,np.newaxis]


train_text=feature_text
train_text_5=np.concatenate((strong_train_data[3],weak_train_data[3]))
train_y=np.concatenate((strong_train_data[2],weak_train_data[2]))



def GenerateLayerTwoTestData():
    strong_test_data=GenerateFromTextToNumpy(1,strong_test)
    weak_test_data=GenerateFromTextToNumpy(0,weak_test)
    test_x={"con":np.concatenate((strong_test_data[0],weak_test_data[0]))[:,:,np.newaxis],"text":np.concatenate((strong_test_data[1],weak_test_data[1])),"text_5":np.concatenate((strong_test_data[3],weak_test_data[3]))}
    test_y=np.concatenate((strong_test_data[2],weak_test_data[2]))
    return test_x,test_y

test_x,test_y=GenerateLayerTwoTestData()


def self_evaluate(x,y):
    res=tf.nn.sigmoid(model.predict(x))  
    res=np.array(res)
    res[res>=0.5]=1
    res[res<0.5]=0
    sum=0
    correct=0
    for i,j in zip(res,y):
        sum+=1
        if i==j:
            correct+=1
    return correct/sum

def load_model():
        global encoder
        kernel_num=32
        input_con=tf.keras.Input(shape=(200,1),name='con')
        y=tf.keras.layers.Conv1D(kernel_num,kernel_size=3,strides=1,padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_con)
        y=tf.keras.layers.Conv1D(kernel_num, kernel_size=3,strides=1, padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)
        
        block1_output=tf.keras.layers.BatchNormalization()(y)

        y=tf.keras.layers.Conv1D(kernel_num, kernel_size=3,strides=1, padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(block1_output)
        y=tf.keras.layers.Conv1D(kernel_num, kernel_size=3,strides=1, padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)

        y=tf.keras.layers.BatchNormalization()(y)
        
        block2_output=tf.keras.layers.add([y,block1_output])

        y=tf.keras.layers.Conv1D(kernel_num, kernel_size=3, padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(block2_output)
        y=tf.keras.layers.Conv1D(kernel_num, kernel_size=3, padding='same', activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01))(y)
        y=tf.keras.layers.BatchNormalization()(y)
        
        block3_output=tf.keras.layers.add([y,block2_output])
        



        y=tf.keras.layers.GlobalAveragePooling1D()(block3_output)
        y=tf.keras.layers.Dense(64,activation='relu')(y)
 
        input_text=tf.keras.Input(shape=(1,),dtype='string',name="text")
        encoder=tf.keras.layers.TextVectorization(max_tokens=100)
        encoder.adapt(train_text)
        x=encoder(input_text)

        x=tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),output_dim=64,mask_zero=True)(x)
        x=tf.keras.layers.Attention(name="ATT12")([x,x])
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True,name="LSTM12"),name="B11")(x)
        x=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,name="LSTM21"),name="B12")(x)
        x=tf.keras.layers.Dense(8,activation='relu')(x)

        input_text_5=tf.keras.Input(shape=(1,),dtype='string',name="text_5")
        encoder_5=tf.keras.layers.TextVectorization(max_tokens=100)
        encoder_5.adapt(train_text_5)
        z=encoder_5(input_text_5)

        z=tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),output_dim=64,mask_zero=True)(z)
        z=tf.keras.layers.Attention(name="ATT1")([z,z])
        z=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True,name="LSTM1"),name="B1")(z)
        z=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,name="LSTM2"),name="B2")(z)
        z=tf.keras.layers.Dense(8,activation='relu')(z)

        #x=tf.keras.layers.add([z,x])

        feature_layer=tf.keras.layers.concatenate([z,y])
        att=tf.keras.layers.Attention()([feature_layer,feature_layer,feature_layer])

        d=tf.keras.layers.Dense(768,activation='relu')(att)
        d=tf.keras.layers.Dropout(0.001)(d)
        output=tf.keras.layers.Dense(1)(d)
        model=tf.keras.Model([input_text,input_con,input_text_5],output)
        base_learning_rate = 0.001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
        return model

po_test=test_x.copy()
po_test["con"]=po_test["con"][0:100]
po_test["text"]=po_test["text"][0:100]
po_test["text_5"]=po_test["text"][0:100]
ne_test=test_x.copy()
ne_test["con"]=ne_test["con"][100:200]
ne_test["text"]=ne_test["text"][100:200]
ne_test["text_5"]=ne_test["text_5"][100:200]

def evaluate(x,y):
    res=model.predict(x)
    t=np.array(tf.nn.sigmoid(res),dtype="float32")
    co=0
    num=y.shape[0]
    for i in range(num):
        if int(t[i][0]+0.5)==y[i]:
            co+=1
    return co/num

def returnAccuracy2():
    import math
    TP=self_evaluate(po_test,test_y[0:100])
    TN=self_evaluate(ne_test,test_y[100:])
    FP=1-TN
    FN=1-TP
    SN=TP/(TP+FN)
    SP=TN/(TN+FP)
    ACC=(TP+TN)/(TP+TN+FN+FP)
    try:
        MCC=(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    except:
        MCC=-1
    text={"ACC":ACC,"SP":SP,"SN":SN,"MCC":MCC}
    return text

kf=KFold(n_splits=10,shuffle=True,random_state=5)
t=kf.split(train_text)
index_list=[(i[0],i[1]) for i in t]

def train(index):    
    epochs=130
    acc=0
    loss=100
    text={}
    t_acc=0
    for i in range(epochs):
        if i<50:
            lr=0.001
        elif i<100:
            lr=0.0005
        else:
            lr=0.0001
            
        def scheduler(epoch):
            return lr
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        print(f"index:{index},epochs:{i}")
        history=model.fit(each_fold_train_x,each_fold_train_y,validation_data=(each_fold_val_x,each_fold_val_y),shuffle=True,epochs=1,batch_size=128,callbacks=[reduce_lr])
        
        train_acc=self_evaluate(each_fold_train_x,each_fold_train_y)
        log_info=history.history
        val_acc=self_evaluate(each_fold_val_x,each_fold_val_y)
        log_info['val_accuracy']=[val_acc]
        log_info['accuracy']=[train_acc]
        print("Training:")
        print(log_info)
        t=returnAccuracy2()
        if i>=0:
            if (history.history["val_accuracy"][0]>=acc and history.history["val_loss"][0]<=loss) or history.history["accuracy"][0]>=t_acc:
                acc=history.history["val_accuracy"][0]
                loss=history.history["val_loss"][0]
                t_acc=history.history["accuracy"][0]
                text=t
                text["val_acc"]=val_acc
                print("UpdateTest:")
                print(text) 
    record.append(text) 



def printFinalResult():
    ACC=0
    SP=0
    SN=0
    MCC=0
    for i in rrecord:
        ACC+=i["ACC"]
        SP+=i["SP"]
        SN+=i["SN"]
        MCC+=i["MCC"]
    ACC/=10
    SP/=10
    SN/=10
    MCC/=10
    text={"ACC":ACC,"SP":SP,"SN":SN,"MCC":MCC}
    print(text)


if __name__=="__main__":
    record=[]
    for fold in range(0,10):
        fold_order=fold
        (train_index,val_index)=index_list[fold_order]
        each_fold_train_x={"con":train_con[train_index],"text":train_text[train_index],"text_5":train_text_5[train_index]}
        each_fold_train_y=train_y[train_index]
        each_fold_val_x={"con":train_con[val_index],"text":train_text[val_index],"text_5":train_text_5[val_index]}
        each_fold_val_y=train_y[val_index]
        model=load_model()
        train(fold)
    printFinalResult()