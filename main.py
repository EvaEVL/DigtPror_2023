from DataPrep import *
from CatBoostModel import *
import pandas as pd


def train():
    data_path = ''
    data = prep_data(data_path)

    X_train, X_test, y_train, y_test = split_for_train(data)

    model = CatModel()
    model.train(X_train, X_test, y_train, y_test)

    return model


def init_pretrained_cat_model():
    model = CatModel()
    model.load_model('Trained_cat_model')
    return model

def make_predict(model, data_to_predict):
    print('Пытаемся предсказать')
    data_to_predict = prepare_for_inference(data_to_predict)
    predict = model.model.predict(data_to_predict)

    return predict


def main():
    data_to_predict = pd.read_excel('to_predict.xlsx')
    data_to_predict_inf = data_to_predict.copy()
    print(data_to_predict.columns)
    print(data_to_predict.head())

    model = init_pretrained_cat_model()
    predicts = make_predict(model, data_to_predict)

    reconvert = {0: "0-7", 1: "8-14", 2 : "15-31",3 : "32-93",4 : "94-186",5 : "186-366", 6: ">366",
                 -1: "-(0-7)", -2: "-(8-14)", -3: "-(15-31)", -4: "-(32-93)", -5: "-(94-186)", -6: "-(186-366)", -7: "< -366"}
    print('Предикты')
    print(predicts.flatten())

    predicts = list(map(lambda x: reconvert[x], predicts.flatten()))


    print("Столбцы")
    print(data_to_predict.columns)

    verds = pd.DataFrame({'Кодзадачи' : data_to_predict_inf['Кодзадачи'], 'Название задачи': data_to_predict_inf['НазваниеЗадачи'],
                         'Кол-во дней': predicts})

    verds.to_csv('predicts.csv')


if __name__ == '__main__':
    main()