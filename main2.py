

from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
import pandas as pd
import uvicorn
import pickle
import numpy as np

scaler = MinMaxScaler()
model2 = pickle.load(open("fd2.pkl", "rb"))
app = FastAPI(title= "PREDICTING REMAINING USEFUL LIFE OF JET ENGINE")
templates = Jinja2Templates(directory="templates")


@app.get("/fd2")
async def form_post(request: Request):
    output = "upload file"
    return templates.TemplateResponse('home2.html.jinja', context={'request': request, 'result': output})


@app.post("/fd2")
async def upload2(request: Request, file2: UploadFile = File(...)):

    contents2 = file2.file.read()
    buffer2 = BytesIO(contents2)
    test2 = pd.read_csv(buffer2)
    buffer2.close()
    file2.file.close()

    drop_col2 = ["slno", "os1", "os2", "os3", "sm16"]
    test2.drop(columns=drop_col2, inplace=True)
    test2_pre = test2.drop_duplicates(["engine"], keep="last")

    x_test2 = scaler.fit_transform(test2_pre.drop(columns=['engine', 'cycles']))
    predictions2 = model2.predict(x_test2)
    li2 = [int(x) for x in predictions2]
    test2 = dict(zip(np.arange(1, 260), li2))

    return templates.TemplateResponse('home2.html.jinja', context={'request': request, 'result': test2})


if __name__ == "__main__":
    uvicorn.run("main2:app", reload=True)

