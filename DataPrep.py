import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import load




def convert_date(diffs):
    """
    Функция для перевода таргета - количество денй между фактическим и планируемым окончанием

    :param diffs:
    :return: category
    """

    if 0<= diffs <= 7:
        return 0
    if 8 <= diffs <= 14:
        return 1
    if 15 <= diffs <= 31:
        return 2
    if 32 <= diffs <= 93:
        return 3
    if 94 <= diffs <= 186:
        return 4
    if 186 <= diffs <= 366:
        return 5
    if diffs > 366:
        return 6

    if 0 > diffs >= -7:
        return -1
    if -7 > diffs >= -15:
        return -2
    if -15 > diffs >= -31:
        return -3
    if -31 > diffs >= -93:
        return -4
    if -93 > diffs >= -186:
        return -5
    if -186 > diffs >= -366:
        return -6
    if diffs < -366:
        return -7


def prep_data(data_path):
    data = pd.read_csv(data_path)

    # Переводим строки дат в формат
    data['ДатаОкончанияЗадачи'] = data['ДатаОкончанияЗадачи'].apply(pd.to_datetime)
    data['ДатаокончанияБП0'] = data['ДатаокончанияБП0'].apply(pd.to_datetime)
    data['ДатаНачалаЗадачи'] = data['ДатаНачалаЗадачи'].apply(pd.to_datetime)
    data['ДатаначалаБП0'] = data['ДатаначалаБП0'].apply(pd.to_datetime)

    # Добавляем тарегт
    data['datediff'] = (data['ДатаОкончанияЗадачи'] - data['ДатаокончанияБП0']).dt.days

    # Избавляемся от лишнего столбца
    data.drop(columns=['Unnamed: 0'], inplace=True)

    # Изменяем дату на альтернативные признаки для дальнейшего обучения
    data['fact_month'] = data['ДатаНачалаЗадачи'].apply(lambda x: x.month)
    data['bpo_month'] = data['ДатаначалаБП0'].apply(lambda x: x.month)
    data['fact_day'] = data['ДатаНачалаЗадачи'].apply(lambda x: x.day)
    data['bpo_day'] = data['ДатаначалаБП0'].apply(lambda x: x.day)

    # Добавляем еще один признак время планируемой работы по докумантам
    data['bpo_diff'] = (data['ДатаокончанияБП0'] - data['ДатаначалаБП0']).dt.days

    # Переводим численные значения разницы по дате в категориальные признаки
    data['converted_datediff'] = data['datediff'].apply(convert_date)

    return data


def split_for_train(data):

    X = data[['obj_subprg', 'obj_key', 'Кодзадачи', 'НазваниеЗадачи',
            'ПроцентЗавершенияЗадачи', 'fact_month', 'bpo_month', 'fact_day', 'bpo_day', 'bpo_diff']]
    y = data['converted_datediff']


    label_encoder_obj_subprg = LabelEncoder()
    X['obj_subprg'] = label_encoder_obj_subprg.fit_transform(X['obj_subprg'])

    label_encoder_obj_key = LabelEncoder()
    X['obj_key'] = label_encoder_obj_key.fit_transform(X['obj_key'])

    label_encoder_task_code = LabelEncoder()
    X['Кодзадачи'] = label_encoder_task_code.fit_transform(X['Кодзадачи'])

    label_encoder_task_name = LabelEncoder()
    X['НазваниеЗадачи'] = label_encoder_task_name.fit_transform(X['НазваниеЗадачи'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12, stratify=y)

    return X_train, X_test, y_train, y_test


def prepare_for_inference(data):
    data.drop(columns=['№ п/п', 'Статуспоэкспертизе', 'Экспертиза'], inplace=True)

    data['ДатаОкончанияЗадачи'] = data['ДатаОкончанияЗадачи'].apply(pd.to_datetime)
    data['ДатаокончанияБП0'] = data['ДатаокончанияБП0'].apply(pd.to_datetime)
    # data['ДатаНачалаЗадачи'] = data['ДатаНачалаЗадачи'].apply(pd.to_datetime)
    data['ДатаначалаБП0'] = data['ДатаначалаБП0'].apply(pd.to_datetime)

    # Добавляем еще один признак время планируемой работы по докумантам
    data['bpo_diff'] = (data['ДатаокончанияБП0'] - data['ДатаначалаБП0']).dt.days

    # Переводим численные значения разницы по дате в категориальные признаки
    # data['converted_datediff'] = data['datediff'].apply(convert_date)

    # Подготовка признаков

    data['fact_month'] = data['ДатаНачалаЗадачи'].apply(lambda x: x.month)
    data['bpo_month'] = data['ДатаначалаБП0'].apply(lambda x: x.month)
    data['fact_day'] = data['ДатаНачалаЗадачи'].apply(lambda x: x.day)
    data['bpo_day'] = data['ДатаначалаБП0'].apply(lambda x: x.day)
    data['bpo_diff'] = (data['ДатаокончанияБП0'] - data['ДатаначалаБП0']).dt.days

    label_encoder_obj_subprg = LabelEncoder()
    label_encoder_obj_subprg = load('label_encoder_obj_subprg.joblib')
    data['obj_subprg'] = data['obj_subprg'].apply(lambda x: 'Общеобразовательные учреждения' if x  not in label_encoder_obj_subprg.classes_ else x)
    data['obj_subprg'] = label_encoder_obj_subprg.transform(data['obj_subprg'])



    label_encoder_obj_key = LabelEncoder()
    label_encoder_obj_key = load('label_encoder_obj_key.joblib')
    data['obj_key'] = data['obj_key'].apply(
        lambda x: '019-0463' if x not in label_encoder_obj_key.classes_ else x)
    data['obj_key'] = label_encoder_obj_key.transform(data['obj_key'])


    label_encoder_task_code = LabelEncoder()
    label_encoder_task_code = load('label_encoder_task_code.joblib')

    data['Кодзадачи'] = data['Кодзадачи'].apply(
        lambda x: '9.6' if x not in label_encoder_task_code.classes_ else x)

    data['Кодзадачи'] = label_encoder_task_code.fit_transform(data['Кодзадачи'])


    label_encoder_task_name = LabelEncoder()
    label_encoder_task_name = load('label_encoder_task_name.joblib')
    data['НазваниеЗадачи'] = data['НазваниеЗадачи'].apply(
        lambda x: 'Получение ответа' if x not in label_encoder_task_code.classes_ else x)
    data['НазваниеЗадачи'] = label_encoder_task_name.transform(data['НазваниеЗадачи'])

    print(data.columns)

    data_to_pred = data[['obj_subprg', 'obj_key', 'Кодзадачи', 'НазваниеЗадачи',
       'ПроцентЗавершенияЗадачи', 'fact_month', 'bpo_month', 'fact_day', 'bpo_day' , 'bpo_diff']]


    return data_to_pred
