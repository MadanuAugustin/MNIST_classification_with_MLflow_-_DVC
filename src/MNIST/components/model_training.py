

from src.MNIST.entity.config_entity import TrainingConfig
import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator


class Training:
    def __init__(self , config : TrainingConfig):
        self.config = config


    # the below function loads the custom updated base model from artifacts
        

    def get_updated_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    ############################################################################################################

    # the below function generates new data from the existing train data 
          
    data_augmentation = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = 0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    ################################################################################################################

    # the below function generates train data
    train_generator = data_augmentation.flow_from_directory(
        # train_data_dir,
        target_size =self.config.params_image_size,
        batch_size = self.config.params_batch_size,
        subset='training',
        shuffle=True,
    )

    ##################################################################################################################

    # the below function generates test data
    test_generator = data_augmentation.flow_from_directory(
        # train_data_dir,
        target_size = self.config.params_image_size,
        batch_size = self.config.params_batch_size,
        subset = 'validation',
        shuffle=False,
    )

    ##################################################################################################################

    def train(self):
        self.training_steps = self.train_generator.samples // self.config.params_batch_size
        self.validation_steps = self.test_generator.samples // self.config.params_batch_size

        self.model.fit(
            self.train_generator,
            epochs = self.config.params_epochs,
            steps_per_epochs = self.training_steps,
            validation_steps = self.validation_steps,
            validation_data = self.test_generator
        )


        tf.keras.models.save_model(model=self.model, filepath=self.config.trained_model_path)


