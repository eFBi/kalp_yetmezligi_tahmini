
"""

# Kalp Yetmezliği Tahmini

## Veri Setini ve Kütüphaneleri Yükleme
"""

import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt

try:
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
except:
    print("Dosya okuma hatası !!!")
else:
    print("Dosya okuma Başarılı...")

data.head()

"""## Veri Setini İnceleme"""

data.info()

data.describe()

data.isnull().sum()

data.corr()['DEATH_EVENT'].sort_values()

plt.figure(figsize=(15,5),dpi=100)
data.corr()['DEATH_EVENT'].sort_values().plot()
plt.title('Ölüm Olaylarının Diğer Özelliklere Göre Korelasyonu')
plt.grid()

kolerasyon = data[['age', 'anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure',
        "serum_creatinine","serum_sodium","sex","smoking","time","DEATH_EVENT",'platelets']]
       
plt.figure(figsize=(20, 8))
sbn.heatmap(kolerasyon.corr(),annot = True,  cmap='RdYlGn_r', mask=np.triu(np.ones_like(kolerasyon.corr())));
plt.title('Correlations between factors', fontsize=20, fontweight='bold',pad=20 );

plt.figure(figsize=(15,4),dpi=100)
plt.subplot(1,2,1)
sbn.distplot(data['serum_creatinine'])
plt.subplot(1,2,2)
sbn.distplot(data['ejection_fraction'])

plt.figure(figsize=(10,4),dpi=100)

sbn.distplot(data['time'])

plt.figure(figsize=(15,8),dpi=100)
plt.subplot(2,2,1)
sbn.scatterplot(x='time',y='serum_creatinine', data=data,hue='DEATH_EVENT')
plt.subplot(2,2,2)
sbn.scatterplot(x='ejection_fraction',y='serum_creatinine', data=data,hue='DEATH_EVENT')
plt.subplot(2,2,3)
sbn.scatterplot(x='ejection_fraction',y='time', data=data,hue='DEATH_EVENT')
plt.subplot(2,2,4)
sbn.scatterplot(x='age',y='time', data=data,hue='DEATH_EVENT')
plt.show()

"""## Veriyi Temizleme"""

sbn.boxplot(x = data.ejection_fraction, color = 'blue')
plt.show()

"""70 ten sonra iki tane verimiz overfitting bu yüzden sileceğiz."""

data[data['ejection_fraction']>=70]

data = data[data['ejection_fraction']<70]

sbn.boxplot(x=data.time, color = 'yellow')
plt.show()

"""Burada overfitting oluşturan bir verimiz yok."""

sbn.boxplot(x=data.serum_creatinine, color = 'red')
plt.show()

"""Burada birçok outliear bulduk ,hepsini önce silerek modelimizi eğittik daha sonra birde silmeden eğittik silmeden eğittiğimiz model daha doğru sonuç verdi."""

#data[data['serum_creatinine']>=1.6].count()

#df = df[df['serum_creatinine']<6]
#sbn.boxplot(x=df.serum_creatinine, color = 'teal')
#plt.show()

"""## Veriyi Test/Train Olarak İkiye Ayırma"""

from sklearn.model_selection import train_test_split

# Y = wX + b

# Y -> Label (Çıktı)
y = data["DEATH_EVENT"].values

# X -> Feature,Attribute (Özellik)
x = data[["ejection_fraction","serum_creatinine","time"]].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=10)

x_train.shape

y_test

"""## Veriyi Normalize Etme (Scaling)"""

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train.shape

x_train[0]

x_test[0]

"""## Modeli Oluşturma"""

import tensorflow as tsf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#np.random.seed(0)

model = Sequential()

model.add(Dense(16,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))

model.add(Dense(1))
"""
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
"""

model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

"""## Modeli Eğitme"""

model.fit(x=x_train,y=y_train,epochs=100,validation_data=(x_test,y_test),verbose=1,callbacks=[early_stopping])

model_loss = pd.DataFrame(model.history.history)

model_loss.plot()

tahminlerimiz = model.predict_classes(x_test)

model.evaluate(x_train,y_train)

train_loss = model.evaluate(x_train,y_train,verbose=0)

test_loss = model.evaluate(x_test,y_test,verbose=0)

print("Eğitim Kayıpları : {} , Test Kayıplsrı : {}".format(train_loss,test_loss))

"""## Modeli Değerlendirme"""

from sklearn.metrics import classification_report , confusion_matrix, accuracy_score

print(classification_report(y_test,tahminlerimiz))

print(confusion_matrix(y_test,tahminlerimiz))

mylist =[]
ac = accuracy_score(y_test,tahminlerimiz)
print("Accuracy")
print(ac)
mylist.append(ac)

"""---

## Farklı bir girdi ile Tahmin Etme
"""

# age 60	creatinine_phosphokinase 581	ejection_fraction 38	serum_creatinine 1.39	serum_sodium 136	time 130
yeni_ornek_ozellikleri = [[40,1,130]]

yeni_ornek_ozellikleri = scaler.transform(yeni_ornek_ozellikleri)

model.predict(yeni_ornek_ozellikleri)
