

from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
import pandas as pd
import uvicorn
import pickle
import numpy as np

scaler = MinMaxScaler()
model1 = pickle.load(open("fd1.pkl", "rb"))
app = FastAPI(title= "PREDICTING REMAINING USEFUL LIFE OF JET ENGINE")
templates = Jinja2Templates(directory="templates")


@app.get("/fd1")
async def form_post(request: Request):
    output = "upload file"
    return templates.TemplateResponse('home1.html.jinja', context={'request': request, 'result': output})


@app.post("/fd1")
async def upload1(request: Request, file1: UploadFile = File(...)):

    contents1 = file1.file.read()
    buffer1 = BytesIO(contents1)
    test1 = pd.read_csv(buffer1)
    buffer1.close()
    file1.file.close()

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
    li1 = [int(x) for x in predictions1]
    test1 = dict(zip(np.arange(1, 101), li1))

    return templates.TemplateResponse('home1.html.jinja', context={'request': request, 'result': test1})


if __name__ == "__main__":
    uvicorn.run("main1:app", reload=True)

