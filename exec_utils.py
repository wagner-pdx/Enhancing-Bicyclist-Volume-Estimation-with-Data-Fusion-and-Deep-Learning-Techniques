from datetime import datetime
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import data_utils

#
#  Run, Evaluate, and Run+Evaluate ALL Configurations
#
def run_and_evaluate_all_configurations(base_dir,
                                        data_dir,
                                        model_configurations):


    #
    #  Create the Results Directory
    #
    results_dir = os.path.join(base_dir, f"Results")
    os.makedirs(results_dir, exist_ok = True)

    #
    #  Prepare and Execute each configuration
    #
    config_count = len(model_configurations)
    for config_idx, model_configuration in enumerate(model_configurations):
        model_str   = model_configuration.get_str()
        config_dict = model_configuration.get_dict()

        w0 = 100
        print("="*w0)
        print("= " + f"Config-[{config_idx+1:03.0f}] of {config_count}".center(w0-4) + " =")
        print("= " + f"{model_str}".center(w0-4) + " =")
        print("="*w0)

        #
        #  Make the Execution Directory for this Configuration.
        #
        config_dir = os.path.join(results_dir, model_str)
        os.makedirs(config_dir, exist_ok = True)

        #
        #  Create the Dataset for Execution
        #
        print(f"Loading 2019-Data...")
        config_dict_2019 = config_dict.copy()
        config_dict_2019['data_year'] = 2019
        input_features_2019, dataset_2019, df_2019, scaler_2019 = data_utils.prepare_data(data_dir, config_dict_2019)

        print(f"Loading 2022-Data...")
        config_dict_2022 = config_dict.copy()
        config_dict_2022['data_year'] = 2022
        input_features_2022, dataset_2022, df_2022, scaler_2022 = data_utils.prepare_data(data_dir, config_dict_2022)

        if config_dict['data_year'] == 2019:
            input_features = input_features_2019
            dataset        = dataset_2019

        if config_dict['data_year'] == 2022:
            input_features = input_features_2022
            dataset        = dataset_2022

        #
        #  Write the input_features, target, and data to the config_dir
        #
        data_utils.write_data(config_dir, config_dict_2019, input_features_2019, dataset_2019, df_2019, scaler_2019)
        data_utils.write_data(config_dir, config_dict_2022, input_features_2022, dataset_2022, df_2022, scaler_2022)


        #
        #  Create the Model for Execution
        #
        model = create_model(model_str, config_dict, config_dir, input_features)

        #
        #  Train the Model (if it hasn't been trained already)
        #
        model_history_filepath = os.path.join(config_dir, "history.txt")
        duration_filepath = os.path.join(config_dir, "duration.txt")
        if os.path.isfile(model_history_filepath) is False:
            #
            #  Train the Model
            #
            training_start_datetime = datetime.now()
            print(f"Start-Time {training_start_datetime}")
            history = train_model(model, dataset, config_dict, config_dir)
            training_end_datetime = datetime.now()
            print(f"End-Time {training_end_datetime}")

            #
            #  Calculate, Print, and Write Duration.
            #
            duration, duration_str = calculate_duration(training_start_datetime, training_end_datetime)
            print(f"Duration {duration}s  --  {duration_str}")
            write_duration(duration, duration_filepath)

            #
            #  Write History
            #
            history = write_history(history, model_history_filepath)

        else:
            #
            #  Read Previous-History and Previous-Duration
            #
            prev_history = read_history(model_history_filepath)
            prev_duration = read_duration(duration_filepath)
                
            #
            #  Load the weights from the most recent/best model-checkpoint.
            #
            best_epoch, checkpoint_filepath = find_best_checkpoint(prev_history, config_dir)
            if best_epoch is not None and checkpoint_filepath is not None:
                model.load_weights(checkpoint_filepath)

            if len(prev_history) < config_dict['epochs']:
                #
                #  Train the Model
                #
                training_start_datetime = datetime.now()
                print(f"Start-Time {training_start_datetime}")
                history = train_model(model, dataset, config_dict, config_dir, latest_epoch=best_epoch)
                training_end_datetime = datetime.now()
                print(f"End-Time {training_end_datetime}")

                #
                #  Calculate, Print, and Write Duration.
                #
                duration_filepath = os.path.join(config_dir, "duration.txt")
                duration, duration_str = calculate_duration(training_start_datetime, training_end_datetime, prev_duration)
                print(f"Duration {duration}s  --  {duration_str}")
                write_duration(duration, duration_filepath)

                #
                #  Merge Histories and Write the Merged-History
                #
                history = merge_histories(history, prev_history)
                history = write_history(history, model_history_filepath)
            else:
                history = prev_history

        #
        #  Create/Write Loss-Plot
        #
        loss_plot_filepath = os.path.join(config_dir, "loss-plot.png")
        data_utils.plot_loss(history, loss_plot_filepath)

        #
        #  Evaluation the Model
        #
        if config_dict['model_type'] == 'DNN':
            if config_dict['data_year'] == 2019:
                results_2019, predictions_2019 = evaluate_DNN_model(model, df_2019, scaler_2019)
                results_2022, predictions_2022 = evaluate_DNN_model(model, df_2022, scaler_2019)
            if config_dict['data_year'] == 2022:
                results_2019, predictions_2019 = evaluate_DNN_model(model, df_2019, scaler_2022)
                results_2022, predictions_2022 = evaluate_DNN_model(model, df_2022, scaler_2022)

        if config_dict['model_type'] == 'LSTM':
            if config_dict['data_year'] == 2019:
                results_2019, predictions_2019 = evaluate_LSTM_model(model, df_2019, scaler_2019)
                results_2022, predictions_2022 = evaluate_LSTM_model(model, df_2022, scaler_2019)
            if config_dict['data_year'] == 2022:
                results_2019, predictions_2019 = evaluate_LSTM_model(model, df_2019, scaler_2022)
                results_2022, predictions_2022 = evaluate_LSTM_model(model, df_2022, scaler_2022)

        #
        #  Write the Predictions for each year in its own file.
        #
        data_utils.write_predictions(config_dir, config_dict_2019, df_2019, predictions_2019)
        data_utils.write_predictions(config_dir, config_dict_2022, df_2022, predictions_2022)

        #
        #  Write the Evaluation Results for each year into its own
        #
        print(f"Results from Evalutation on the 2019-Data")
        data_utils.write_evaluation_results(config_dir, config_dict_2019, results_2019)
        for metric, value in results_2019.items():
            print(f"{metric:<40s} = {value:>12.4f}")
        print()

        print(f"Results from Evalutation on the 2022-Data")
        data_utils.write_evaluation_results(config_dir, config_dict_2022, results_2022)
        for metric, value in results_2022.items():
            print(f"{metric:<40s} = {value:>12.4f}")
        print()

def evaluate_all_configurations(base_dir,
                                data_dir,
                                model_configurations):


    #
    #  Create the Results Directory
    #
    results_dir = os.path.join(base_dir, f"Results")
    os.makedirs(results_dir, exist_ok = True)



    #
    #  Prepare and Execute each configuration
    #
    config_count = len(model_configurations)
    for config_idx, model_configuration in enumerate(model_configurations):
        model_str   = model_configuration.get_str()
        config_dict = model_configuration.get_dict()

        w0 = 100
        print("="*w0)
        print("= " + f"Config-[{config_idx+1:03.0f}] of {config_count}".center(w0-4) + " =")
        print("= " + f"{model_str}".center(w0-4) + " =")
        print("="*w0)

        #
        #  Make the Execution Directory for this Configuration.
        #
        config_dir = os.path.join(results_dir, model_str)
        os.makedirs(config_dir, exist_ok = True)

        #
        #  See if we have a trained model
        #
        model_history_filepath = os.path.join(config_dir, "history.txt")
        if os.path.isfile(model_history_filepath) is False:
            print(f"ERROR: History File Not Found.")
            return False
        
        #
        #  Read Previous-History and Previous-Duration
        #
        history = read_history(model_history_filepath)

        #
        #  Check to see if we have a completed model
        #
        expected_epochs = config_dict['epochs']
        executed_epochs = len(history)
        if executed_epochs < expected_epochs:
            print(f"ERROR: Model not previously executed fully.")
            print(f"     : expected epochs {expected_epochs}")
            print(f"     : executed epochs {executed_epochs}")
            return False

        #
        #  Create the Dataset for Execution
        #
        print(f"Loading 2019-Data...")
        config_dict_2019 = config_dict.copy()
        config_dict_2019['data_year'] = 2019
        input_features_2019, dataset_2019, df_2019, scaler_2019 = data_utils.prepare_data(data_dir, config_dict_2019)

        print(f"Loading 2022-Data...")
        config_dict_2022 = config_dict.copy()
        config_dict_2022['data_year'] = 2022
        input_features_2022, dataset_2022, df_2022, scaler_2022 = data_utils.prepare_data(data_dir, config_dict_2022)

        if config_dict['data_year'] == 2019:
            input_features = input_features_2019
        if config_dict['data_year'] == 2022:
            input_features = input_features_2022

        #
        #  Create the Model for Execution
        #
        model = create_model(model_str, config_dict, config_dir, input_features)

        #
        #  Load the weights from the most recent/best model-checkpoint.
        #
        best_epoch, checkpoint_filepath = find_best_checkpoint(history, config_dir)
        if best_epoch is not None and checkpoint_filepath is not None:
            model.load_weights(checkpoint_filepath)

            val_result = history.loc[best_epoch-1]
            if 'root_mean_squared_error' not in val_result.keys():
                val_result['root_mean_squared_error'] = np.sqrt(val_result['mean_squared_error'])
            if 'val_root_mean_squared_error' not in val_result.keys():
                val_result['val_root_mean_squared_error'] = np.sqrt(val_result['val_mean_squared_error'])


            val_results_filepath = os.path.join(config_dir, f"Validation-Results.csv")
            with open(val_results_filepath, mode='w') as f:
                val_result.to_csv(f)

            val_results_filepath = os.path.join(config_dir, f"Validation-Results.txt")
            with open(val_results_filepath, mode='w') as f:
                for key, value in val_result.items():
                    f.write(f"{key:<40s}: {value:.6f}" + '\n')

        #
        #  Write the input_features, target, and data to the config_dir
        #
        data_utils.write_data(config_dir, config_dict_2019, input_features_2019, dataset_2019, df_2019, scaler_2019)
        data_utils.write_data(config_dir, config_dict_2022, input_features_2022, dataset_2022, df_2022, scaler_2022)

        #
        #  Create/Write Loss-Plot
        #
        loss_plot_filepath = os.path.join(config_dir, "loss-plot.png")
        data_utils.plot_loss(history, loss_plot_filepath)

        #
        #  Evaluation the Model
        #
        if config_dict['model_type'] == 'DNN':
            if config_dict['data_year'] == 2019:
                results_2019, predictions_2019 = evaluate_DNN_model(model, df_2019, scaler_2019)
                results_2022, predictions_2022 = evaluate_DNN_model(model, df_2022, scaler_2019)
            if config_dict['data_year'] == 2022:
                results_2019, predictions_2019 = evaluate_DNN_model(model, df_2019, scaler_2022)
                results_2022, predictions_2022 = evaluate_DNN_model(model, df_2022, scaler_2022)

        if config_dict['model_type'] == 'LSTM':
            if config_dict['data_year'] == 2019:
                results_2019, predictions_2019 = evaluate_LSTM_model(model, df_2019, scaler_2019)
                results_2022, predictions_2022 = evaluate_LSTM_model(model, df_2022, scaler_2019)
            if config_dict['data_year'] == 2022:
                results_2019, predictions_2019 = evaluate_LSTM_model(model, df_2019, scaler_2022)
                results_2022, predictions_2022 = evaluate_LSTM_model(model, df_2022, scaler_2022)

        #
        #  Write the Predictions for each year in its own file.
        #
        data_utils.write_predictions(config_dir, config_dict_2019, df_2019, predictions_2019)
        data_utils.write_predictions(config_dir, config_dict_2022, df_2022, predictions_2022)

        #
        #  Write the Evaluation Results for each year into its own file.
        #
        data_utils.write_evaluation_results(config_dir, config_dict_2019, results_2019)
        data_utils.write_evaluation_results(config_dir, config_dict_2022, results_2022)

        #
        #  Display the Evaluation Results for each year
        #
        print()
        print(f"Results from Evalutation on the 2019-Data")
        for metric, value in results_2019.items():
            print(f"{metric:<40s} = {value:>12.4f}")

        print()
        print(f"Results from Evalutation on the 2022-Data")
        for metric, value in results_2022.items():
            print(f"{metric:<40s} = {value:>12.4f}")

        print()


#
#  Model Functions
#
def create_model(model_str, config_dict, config_dir, input_features):
    layer_count = config_dict['layer_count']
    node_count  = config_dict['node_count']

    layers = []

    assert config_dict['model_type'] == 'DNN' or config_dict['model_type'] == 'LSTM'

    feature_count = len(input_features)
    if config_dict['model_type'] == 'LSTM':
        input_shape = (1, feature_count)
    if config_dict['model_type'] == 'DNN':
        input_shape = (feature_count, )

    #  Input Layer
    input_layer = tf.keras.Input(shape=input_shape, name="input")
    layers.append(input_layer)

    # Deep Layers (LSTM)
    if config_dict['model_type'] == 'LSTM':
        for i in range(layer_count):
            if i+1 == layer_count:
                new_layer = tf.keras.layers.LSTM(node_count, name=f'LSTM_{i:02.0f}', return_sequences=False)
                layers.append(new_layer)
            else:
                new_layer = tf.keras.layers.LSTM(node_count, name=f'LSTM_{i:02.0f}', return_sequences=True)
                layers.append(new_layer)

    # Deep Layers (DNN)
    if config_dict['model_type'] == 'DNN':
        for i in range(layer_count):
            new_layer = tf.keras.layers.Dense(node_count, activation='relu', name=f'Dense_{i:02.0f}')
            layers.append(new_layer)
    
    #  Output Layer
    final_layer = tf.keras.layers.Dense(1, name='final')
    layers.append(final_layer)

    # Define Model
    model_str_name = model_str.replace("(", "_").replace(")", "_").replace("+", "-")
    model = tf.keras.models.Sequential(layers, name=model_str_name)

    # Define Metrics
    metrics = ['mean_absolute_error',
               'mean_absolute_percentage_error', 
               'root_mean_squared_error', 
               'r2_score']

    # Compile Model
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error',
                  metrics=metrics)

    # save summary to folder
    model_summary_filepath = os.path.join(config_dir, "model_summary.txt")
    with open(model_summary_filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

def train_model(model, data, config_dict, config_dir, latest_epoch=None):
    x_train, x_eval, y_train, y_eval = data

    x_train = data_utils.reshape_data(config_dict, x_train.to_numpy())
    x_eval  = data_utils.reshape_data(config_dict, x_eval.to_numpy())

    batch_size = config_dict['batch_size']
    max_epochs = int(config_dict['epochs'])

    checkpoint_dir = os.path.join(config_dir, f"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok = True)

    checkpoint_filepath = os.path.join(checkpoint_dir, "model-checkpoint-{epoch:03d}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, 
                                                             save_weights_only=True, 
                                                             monitor='val_loss', 
                                                             mode='auto', 
                                                             save_best_only=True, 
                                                             verbose=0)

    #
    #  Calculate how many epochs to train for, given our checkpoint
    #
    if latest_epoch is None:
        latest_epoch = 0


    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        initial_epoch=latest_epoch,
                        epochs=max_epochs, 
                        callbacks=[checkpoint_callback],
                        validation_data=(x_eval, y_eval),
                        verbose=1)
    
    #
    #  Remove ALL model-checkpoints, except the best one
    #
    clean_up_checkpoints(config_dir, history)

    return history

def evaluate_DNN_model(model, df, scaler):
    x_df, y_df = df
    x_np = scaler.transform(x_df).reshape((-1, x_df.shape[0], x_df.shape[1]))
    y_np = y_df.to_numpy().reshape((-1, y_df.shape[0]))

    ds = tf.data.Dataset.from_tensor_slices((x_np, y_np))
    results = model.evaluate(ds, return_dict=True)
    predictions = model.predict_on_batch(x_np.reshape((x_df.shape[0], x_df.shape[1])))

    return results, predictions

def evaluate_LSTM_model(model, df, scaler):
    x_df, y_df = df
    x_np = scaler.transform(x_df).reshape((-1, x_df.shape[0], 1, x_df.shape[1]))
    y_np = y_df.to_numpy().reshape((-1, y_df.shape[0]))

    ds = tf.data.Dataset.from_tensor_slices((x_np, y_np))
    results = model.evaluate(ds, return_dict=True)
    predictions = model.predict_on_batch(x_np.reshape((x_df.shape[0], 1, x_df.shape[1])))

    return results, predictions

#
#  Checkpoint Function
#
def find_best_checkpoint(history, config_dir):
    checkpoint_dir = os.path.join(config_dir, f"checkpoints")

    #
    #  Find the iteration with minimum evaluation loss value
    #
    val_loss = np.asarray(history['val_loss'])
    best_epoch = np.argmin(val_loss) + 1 # Model-Checkpoints are One-Indexed

    #
    #  Load the weights from the best iteration into the model
    #
    checkpoint_filepath = os.path.join(checkpoint_dir, "model-checkpoint-{epoch:03d}.weights.h5".format(epoch=best_epoch))
    if os.path.exists(checkpoint_filepath) is False:
        print(f"File Does Not Exist: {checkpoint_filepath}")
        return None, None
    else:
        return best_epoch, checkpoint_filepath

def clean_up_checkpoints(config_dir, history):

    best_epoch, best_checkpoint_filepath = find_best_checkpoint(history.history, config_dir)

    checkpoint_dir = os.path.join(config_dir, "checkpoints")
    checkpoint_pattern = "model-checkpoint-*.weights.h5"
    
    checkpoint_pattern_path = os.path.join(checkpoint_dir, checkpoint_pattern)
    checkpoint_files = glob.glob(checkpoint_pattern_path)

    if best_checkpoint_filepath in checkpoint_files:
        print(f"we found our best checkpoint, and removed it from the chopping block.")
        checkpoint_files.remove(best_checkpoint_filepath)

        assert best_checkpoint_filepath not in checkpoint_files

        for checkpoint_file in checkpoint_files:
            os.remove(checkpoint_file)
    else:
        print(f"we did not find the best checkpoint filepath among those we found in the checkpoints directory.")
        print(f"therefore, we are not removing any checkpoint for now.")

#
#  Duration Functions
#
def calculate_duration(start_datetime, end_datetime, prev_duration=None):
    if prev_duration is None:
        duration = (end_datetime - start_datetime).total_seconds()
    else:
        duration = (end_datetime - start_datetime).total_seconds() + prev_duration
    
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)

    duration_str = f"{h:3.0f}h {m:02.0f}m {s:02.0f}s"

    return duration, duration_str

def write_duration(duration, duration_filepath):
    with open(duration_filepath, mode='w') as f:
        f.write(str(duration) + '\n')

def read_duration(duration_filepath):
    with open(duration_filepath, mode='r') as f:
        read = f.read()
    duration = float(read)
    return duration

#
#  History Functions
#
def merge_histories(history, prev_history_df):
    #  how many epochs did we previously run?
    prev_epochs = len(prev_history_df) # Zero-Indexed

    #  Recall we ran from the last/best checkpoint
    val_loss = np.asarray(prev_history_df['val_loss'])
    best_epoch = np.argmin(val_loss)

    if best_epoch+1 <= prev_epochs-1:
        prev_history_df = prev_history_df.drop(np.arange(best_epoch+1, prev_epochs))

    #  how many new epochs did we run?
    new_epoch_count = len(history.history['val_loss'])
    #  Continue from the previous number of epochs for however many we just ran.
    new_epochs = np.arange(best_epoch+1, best_epoch+1+new_epoch_count)

    #  make the history a dataframe
    history_df = pd.DataFrame(history.history, index=new_epochs)
    history_df.index.rename("epoch")

    #  Merge the two dataframe to create a complete/consistent history
    new_history = pd.concat([prev_history_df, history_df])
    new_history.index.rename("epoch")

    #  Assert that the epochs are contiguous and unbroken from 0 and on
    epochs = new_history.index.values
    for i in range(new_history.shape[0]):
        assert i in epochs

    return new_history

def write_history(history, model_history_filepath):
    if type(history) is not pd.DataFrame:
        history_df = pd.DataFrame(history.history)
        history_df.index.names = ["epoch"]
    else:
        history_df = history
    
    with open(model_history_filepath, mode='w') as f:
        history_df.to_csv(f)
    
    history_df = read_history(model_history_filepath)

    return history_df

def read_history(model_history_filepath):
    history_df = pd.read_csv(model_history_filepath, index_col=0)
    history_df.index.names = ["epoch"]

    return history_df
