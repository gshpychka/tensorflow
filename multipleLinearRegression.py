#%% Imports
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

#%% Set options
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#%% Import the dataset
california_housing_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv",
                                           sep=",")

#%% Shuffle the dataset.
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
#%% def preprocess_features() and preprocess_targets()


def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing dataset.

    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data from the California housing dataset.
    Returns:
        A DataFrame that contains the features to be used for the model, including synthetic features
    """

    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]
    ]
    processed_features = selected_features.copy()

    # Create a synthetic feature
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"]
    )
    return processed_features


def preprocess_targets(california_housing_dataframe):
    """Prepares target features (labels) from California housing dataset

    Args:
         california_housing_dataframe: A Pandas DataFrame expected to contain data from the California housing dataset
    Returns:
        A DataFrame that contains the target feature
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.00)
    return output_targets

#%% Assign training and validation examples and targets


training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

#%% def plot_data_geographically()


def plot_data_geographically(validation_examples, training_examples):

    # Plot Latitude/Longitude vs Median House Value

    plt.figure(figsize=(13, 8))
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] /
                validation_targets["median_house_value"].max())  # range 0-1

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())  # range 0-1

    _ = plt.plot()

#%% Plot data geographically


# plot_data_geographically(validation_examples, training_examples)


# %% Define my_input_fn(), construct_feature_columns()


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

        Args:
          features: pandas DataFrame of features
          targets: pandas DataFrame of targets
          batch_size: Size of batches to be passed to the model
          shuffle: True or False. Whether to shuffle the data.
          num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
        Returns:
          Tuple of (features, labels) for next data batch
        """

    # Convert pandas data into a dict of np arrays
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle if specified
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(training_examples):
    """Construct the TensorFlow Feature Columns.
    Args:
        training_examples: a Dataframe with the training examples
    Returns:
        A set of feature columns
    """
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # Divide households into 7 buckets
    bucketized_households = tf.feature_column.bucketized_column(
        households,
        boundaries=get_quantile_based_boundaries(training_examples["households"], 7))
    bucketized_longtitude = tf.feature_column.bucketized_column(
        longitude,
        boundaries=get_quantile_based_boundaries(training_examples["longitude"], 7))
    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude,
        boundaries=get_quantile_based_boundaries(training_examples["latitude"], 7))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        housing_median_age,
        boundaries=get_quantile_based_boundaries(training_examples["housing_median_age"], 7))
    bucketized_median_income = tf.feature_column.bucketized_column(
        median_income,
        boundaries=get_quantile_based_boundaries(training_examples["median_income"], 7))
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        rooms_per_person,
        boundaries=get_quantile_based_boundaries(training_examples["rooms_per_person"], 7))

    feature_columns = [
        bucketized_longtitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person
    ]

    return feature_columns


#%% Define train_model()
def train_model(
        learning_rate,
        steps,
        batch_size,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

      In addition to training, this function also prints training progress information,
      as well as a plot of the training and validation loss over time.

      Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
          consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing one or more columns from
          `california_housing_dataframe` to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from
          `california_housing_dataframe` to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns from
          `california_housing_dataframe` to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column from
          `california_housing_dataframe` to use as target for validation.

      Returns:
        A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    training_input_fn = lambda: my_input_fn(
        features=training_examples,
        targets=training_targets["median_house_value"],
        batch_size=batch_size
    )

    predict_training_input_fn = lambda: my_input_fn(
        features=training_examples,
        targets=training_targets["median_house_value"],
        num_epochs=1,
        shuffle=False
    )

    predict_validation_input_fn = lambda: my_input_fn(
        features=validation_examples,
        targets=validation_targets["median_house_value"],
        num_epochs=1,
        shuffle=False
    )

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()
    return linear_regressor

#%% Create a correlation dataframe


correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

correlation_dataframe = correlation_dataframe.corr()


# %% def get_quantile_based_boundaries()


def get_quantile_based_boundaries(feature_values, num_buckets):
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


#%% Train with selected features

linear_regressor = train_model(
    learning_rate=1,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)


#%% Train with all features
# linear_regressor = train_model(
#     learning_rate=0.0003,
#     steps=500,
#     batch_size=5,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)
#%%
# california_housing_test_data = pd.read_csv(
#     "https://dl.google.com/mlcc/mledu-datasets/california_housing_test.csv", sep=",")
#
# test_examples = preprocess_features(california_housing_test_data)
# test_targets = preprocess_targets(california_housing_test_data)
#
# predict_test_input_fn = lambda: my_input_fn(
#     features=test_examples,
#     targets=test_targets["median_house_value"],
#     num_epochs=1,
#     shuffle=False
# )
#
# test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
# test_predictions = np.array([item['predictions'][0] for item in test_predictions])
#
# test_root_mean_squared_error = math.sqrt(
#     metrics.mean_squared_error(test_predictions, test_targets))
#
# print("Test RMSE: %0.2f" % test_root_mean_squared_error)
