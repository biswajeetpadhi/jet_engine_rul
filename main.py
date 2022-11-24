from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import FileResponse
from io import BytesIO
import pandas as pd
import numpy as np
import uvicorn
import pickle

scaler = MinMaxScaler()
app = FastAPI()
model1 = pickle.load(open("fd1.pkl", "rb"))
model2 = pickle.load(open("fd2.pkl", "rb"))
model3 = pickle.load(open("fd3.pkl", "rb"))
model4 = pickle.load(open("fd4.pkl", "rb"))


@app.post("/fd1")
def upload(file: UploadFile = File(...)):
    contents1 = file.file.read()
    buffer1 = BytesIO(contents1)
    test1 = pd.read_csv(buffer1)
    buffer1.close()
    file.file.close()

    drop_col1 = ["slno", "os1", "os2", "os3", "sm1", "sm5", "sm6", "sm10", "sm16", "sm18", "sm19"]
    test1.drop(columns=drop_col1, inplace=True)

    data1 = []
    for i in np.arange(1, 101):
        temp_test_data1 = test1[test1['engine'] == i].values
        data1.append(temp_test_data1[-1])

    coln1 = list(test1.columns)
    test1_pre = pd.DataFrame(data1, columns=coln1)

    x_test1 = scaler.fit_transform(test1_pre.drop(columns=['engine', 'cycles']))
    predictions1 = model1.predict(x_test1)

    df1 = pd.Series(predictions1)
    df1.to_csv("df1.csv")

    return FileResponse("df1.csv")


@app.post("/fd2")
def upload(file: UploadFile = File(...)):
    contents2 = file.file.read()
    buffer2 = BytesIO(contents2)
    test2 = pd.read_csv(buffer2)
    buffer2.close()
    file.file.close()

    drop_col2 = ["slno", "os1", "os2", "os3", "sm16"]
    test2.drop(columns=drop_col2, inplace=True)

    data2 = []
    for i in np.arange(1, 260):
        temp_test_data2 = test2[test2['engine'] == i].values
        data2.append(temp_test_data2[-1])

    coln2 = list(test2.columns)
    test2_pre = pd.DataFrame(data2, columns=coln2)

    x_test2 = scaler.fit_transform(test2_pre.drop(columns=['engine', 'cycles']))
    predictions2 = model2.predict(x_test2)

    df2 = pd.Series(predictions2)
    df2.to_csv("df2.csv")

    return FileResponse("df2.csv")


@app.post("/fd3")
def upload(file: UploadFile = File(...)):
    contents3 = file.file.read()
    buffer3 = BytesIO(contents3)
    test3 = pd.read_csv(buffer3)
    buffer3.close()
    file.file.close()

    drop_col3 = ["slno", "os1", "os2", "os3", "sm1", "sm5", "sm6", "sm10", "sm16", "sm18", "sm19"]
    test3.drop(columns=drop_col3, inplace=True)

    data3 = []
    for i in np.arange(1, 101):
        temp_test_data3 = test3[test3['engine'] == i].values
        data3.append(temp_test_data3[-1])

    coln3 = list(test3.columns)
    test3_pre = pd.DataFrame(data3, columns=coln3)

    x_test3 = scaler.fit_transform(test3_pre.drop(columns=['engine', 'cycles']))
    predictions3 = model3.predict(x_test3)

    df3 = pd.Series(predictions3)
    df3.to_csv("df3.csv")

    return FileResponse("df3.csv")


@app.post("/fd4")
def upload(file: UploadFile = File(...)):
    contents4 = file.file.read()
    buffer4 = BytesIO(contents4)
    test4 = pd.read_csv(buffer4)
    buffer4.close()
    file.file.close()

    drop_col4 = ["slno", "os1", "os2", "os3", "sm16"]
    test4.drop(columns=drop_col4, inplace=True)

    data4 = []
    for i in np.arange(1, 249):
        temp_test_data4 = test4[test4['engine'] == i].values
        data4.append(temp_test_data4[-1])

    coln4 = list(test4.columns)
    test4_pre = pd.DataFrame(data4, columns=coln4)

    x_test4 = scaler.fit_transform(test4_pre.drop(columns=['engine', 'cycles']))
    predictions4 = model4.predict(x_test4)

    df4 = pd.Series(predictions4)
    df4.to_csv("df4.csv")

    return FileResponse("df4.csv")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

