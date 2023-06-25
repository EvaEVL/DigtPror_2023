import catboost as cb

class CatModel():

    def __init__(self):
        self.model = cb.CatBoostClassifier(
            random_seed=12,
            learning_rate=0.05,
            iterations=200,
            # task_type="CPU",
            loss_function='MultiClass',
            custom_loss=['Accuracy'],
        )

    def train(self, X_train, X_test, y_train, y_test):
        self.model.fit(
            X_train, y_train,
            # cat_features=cat_features,
            eval_set=(X_test, y_test),
            # eval_metric='Accuracy',
            logging_level='Verbose',
            # verbose=True,
            plot=True,
    )

    def load_model(self, module_path):
        self.model = self.model.load_model(module_path)

