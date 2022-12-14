

from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
import pandas as pd
import uvicorn
import pickle
import numpy as np

scaler = MinMaxScaler()
model3 = pickle.load(open("fd3.pkl", "rb"))
app = FastAPI(title= "PREDICTING REMAINING USEFUL LIFE OF JET ENGINE")
templates = Jinja2Templates(directory="templates")


@app.get("/fd3")
async def form_post(request: Request):
    output = "upload file"
    return templates.TemplateResponse('home3.html.jinja', context={'request': request, 'result': output})



@app.post("/fd3")
async def upload3(request: Request, file3: UploadFile = File(...)):

    contents3 = file3.file.read()
    buffer3 = BytesIO(contents3)
    test3 = pd.read_csv(buffer3)
    buffer3.close()
    file3.file.close()

    drop_col3 = ["slno", "os1", "os2", "os3", "sm1", "sm5", "sm6", "sm10", "sm16", "sm18", "sm19"]
    test3.drop(columns=drop_col3, inplace=True)
    test3_pre = test3.drop_duplicates(["engine"], keep="last")

    x_test3 = scaler.fit_transform(test3_pre.drop(columns=['engine', 'cycles']))
    predictions3 = model3.predict(x_test3)
    li3 = [int(x) for x in predictions3]
    test3 = dict(zip(np.arange(1, 260), li3))

    return templates.TemplateResponse('home3.html.jinja', context={'request': request, 'result': test3})


if __name__ == "__main__":
    uvicorn.run("main3:app", reload=True)

