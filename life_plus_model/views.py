import pickle
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response

model = pickle.load(open('life_plus_model/model.sav', 'rb'))
scaler = pickle.load(open('life_plus_model/std_scaler.bin', 'rb'))


@api_view(['POST'])
def model_view(request):
    if model is None or scaler is None:
        loadModel()
    # scaled, result = processRequest(63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)
    body = request.data
    age = body["age"]
    sex = body["sex"]
    chest_pain = body["chestPain"]
    trestbps = body["trestbps"]
    chol = body["chol"]
    fbs = body["fbs"]
    restecg = body["restecg"]
    thalach = body["thalach"]
    exang = body["exang"]
    oldpeak = body["oldpeak"]
    slope = body["slope"]
    ca = body["ca"]
    thal = body['thal']
    # scaled, result = processRequest(56,1,0,130,283,1,0,103,1,1.6,0,0,3)
    result = processRequest(age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if result[0] == 0:
        body = {"hasDisease": True}
    else:
        body = {"hasDisease": False}

    return JsonResponse(body)


    #              age, sex, cp,         trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
def processRequest(age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    #  0     1       2     3       4     5   6      7     8    9   10      11       12     13     14     15     16    17   18   19   20
    # age,trestbps,chol,thalach,oldpeak,sex,fbs,restecg,exang,ca,slope_0,slope_1,slope_2,thal_0,thal_1,thal_2,thal_3,cp_0,cp_1,cp_2,cp_3
    basic = [sex, fbs, restecg, exang, ca, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    toBeScaled = [age, trestbps, chol, thalach, oldpeak]
    scaled = scaler.transform(np.reshape(toBeScaled, (1, -1)))[0]
    print(scaled)
    basic = np.append(scaled, basic)

    slope_0_index = 10
    thal_0_index = 13
    cp_0_index = 17

    assert (chest_pain >= 0 and chest_pain <= 3)
    basic[cp_0_index + chest_pain] = 1
    assert (thal >= 0 and thal <= 3)
    basic[thal_0_index + thal] = 1
    assert (slope >= 0 and slope <= 2)
    basic[slope_0_index + slope] = 1
    return model.predict([basic])


def loadModel():
    global model
    global scaler
    scaler = pickle.load(open('std_scaler.bin', 'rb'))
    model = pickle.load(open('model.sav', 'rb'))
