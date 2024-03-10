

from src.MNIST.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config : PrepareBaseModelConfig):
        self.config = config

    # the below function is responsible for creating the base model vgg16
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
                include_top=self.config.params_include_top,
                weights=self.config.params_weights,
                input_shape=self.config.params_image_size,
                classes=self.config.params_classes,          
        )

        # self.save_model(path = self.config.base_model_path, model = self.model)

        tf.keras.models.save_model(model=self.model, filepath=self.config.base_model_path)



    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # we are not going to train the layers of the model if freeze_all is true
        if freeze_all:
            for  layer in model.layers:
                layer.trainable = False
        # we are not going to train the layers of the model upto the freeze_till layer
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # creating the flatten layer after vgg16 last layer
        
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # creating the hidden layers and output layers

        predicition = tf.keras.layers.Dense(
            units = classes,
            activation = 'softmax'
        )(flatten_in)


        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = predicition
        )

        optimizer = tf.keras.optimizers.SGD(learning_rate= learning_rate)


        full_model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy']
        )

        full_model.summary()

        return full_model
    



    # the below method is responsible for updating the base model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model = self.model, ## this is the base model
            classes = self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        # self.save_model(path = self.config.updated_base_model_path , model = self.full_model)
        
        tf.keras.models.save_model(model=self.full_model, filepath=self.config.updated_base_model_path)




    # @staticmethod
    # def save_model(path : Path, model : tf.keras.Model):
    #     model.save(path)
