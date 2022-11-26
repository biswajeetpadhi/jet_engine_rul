


from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, File, UploadFile, Request
from io import BytesIO
import pandas as pd
import uvicorn
import pickle
import numpy as np

scaler = MinMaxScaler()
model4 = pickle.load(open("fd4.pkl", "rb"))
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/fd4")
async def form_post(request: Request):
    output = "upload file"
    return templates.TemplateResponse('home4.html', context={'request': request, 'result': output})



@app.post("/fd4")
async def upload4(request: Request, file4: UploadFile = File(...)):

    contents4 = file4.file.read()
    buffer4 = BytesIO(contents4)
    test4 = pd.read_csv(buffer4)
    buffer4.close()
    file4.file.close()

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

    li4 = [int(x) for x in predictions4]
    test4 = dict(zip(np.arange(1, 249), li4))

    return templates.TemplateResponse('home4.html', context={'request': request, 'result': test4})


if __name__ == "__main__":
    uvicorn.run("main4:app", reload=True)

