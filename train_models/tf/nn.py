import constants
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from models.model import Model

class FCNet(Model):
    def __init__(
            self,
            input_shape: tuple,
            random_seed: int = 0):
        super().__init__(input_shape=input_shape, random_seed=random_seed)

        self._epochs = None
        self._batch_size = None
        self._early_stopping_epochs = None
        self._learning_rate_decay_factor = None
        self._learning_rate_decay_epochs = None

    def get_model_name(self) -> str:
        return constants.NN_MODEL_NAME

    def _build_model(self, **kwargs):
        self._epochs = kwargs[constants.EPOCHS]
        self._batch_size = kwargs[constants.BATCH_SIZE]
        self._early_stopping_epochs = kwargs[constants.EARLY_STOPPING_EPOCHS]
        self._learning_rate_decay_factor = kwargs[constants.LEARNING_RATE_DECAY_FACTOR]
        self._learning_rate_decay_epochs = kwargs[constants.LEARNING_RATE_DECAY_EPOCHS]

        tf.random.set_seed(seed=self.random_seed)

        noise_range = kwargs[constants.NOISE_RANGE]
        hidden_layers = kwargs[constants.HIDDEN_LAYERS]
        activations = kwargs[constants.ACTIVATIONS]
        batch_normalizations = kwargs[constants.BATCH_NORMALIZATIONS]
        regularizations = kwargs[constants.REGULARIZATIONS]
        dropouts = kwargs[constants.DROPOUTS]
        optimizer = kwargs[constants.OPTIMIZER]
        learning_rate = kwargs[constants.LEARNING_RATE]

        match optimizer:
         case constants.OPTIMIZER_ADAM:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
         case constants.OPTIMIZER_YOGI:
            optimizer = tfa.optimizers.Yogi(learning_rate=learning_rate)
         case constants.OPTIMIZER_ADAMW:
            optimizer = tfa.optimizers.AdamW(weight_decay=0.001, learning_rate=learning_rate)
         case _:
            raise NotImplementedError(f'Optimizer "{optimizer}" has not been implemented yet')

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))

        if noise_range > 0.0:
            model.add(tf.keras.layers.GaussianNoise(stddev=noise_range))

        for i, units in enumerate(hidden_layers):
            regularizer = regularizations[i]
            batch_norm = batch_normalizations[i]
            dropout = dropouts[i]
            model.add(tf.keras.layers.Dense(
                    units=units,
                    activation=activations[i],
                    use_bias=not batch_norm,
                    kernel_regularizer=regularizer))
            
            if batch_normalizations[i]:
                model.add(tf.keras.layers.BatchNormalization())
            if dropout > 0.0:
                model.add(tf.keras.layers.Dropout(rate=dropout))

        model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model

    def save(self, checkpoint_filepath: str):
        self.model.save(checkpoint_filepath)

    def _load(self, checkpoint_filepath: str):
        return tf.keras.models.load_model(checkpoint_filepath)

    def _train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_test: np.ndarray,
            y_test: np.ndarray):
        callbacks = []
        if self._early_stopping_epochs > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self._early_stopping_epochs,
                restore_best_weights=True))
        if self._learning_rate_decay_epochs > 0 and self._learning_rate_decay_factor > 0.0:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=self._learning_rate_decay_factor,
                patience=self._learning_rate_decay_epochs))

        self.model.fit(
            x_train,
            y_train,
            batch_size=self._batch_size,
            epochs=self._epochs,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            verbose=1)

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        predict_proba = np.round(self.model.predict(x), 2)
        y_pred = np.argmax(predict_proba, axis=1)
        return y_pred, predict_proba
