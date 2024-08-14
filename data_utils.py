import pickle
import time
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from feature_sets import key_features
from feature_sets import daily_features
from feature_sets import monthly_features
from feature_sets import daily_strava_features
from feature_sets import monthly_strava_features
from feature_sets import static_features

#
#  Load-Data
#
def load_data(data_dir, config_dict):
    model_type  = config_dict['model_type']
    feature_set = config_dict['feature_set']
    target      = config_dict['target']
    data_year   = config_dict['data_year']

    #
    #  Pick the right input-files based on data_year
    #
    if data_year == 2019:
        static_data_filepath         = os.path.join(data_dir, f"data/static/df1-2019.csv")
        monthly_strava_data_filepath = os.path.join(data_dir, f"data/counts/combined/output/combined-pc-counts-monthly-stv2019.csv")
        daily_strava_data_filepath   = os.path.join(data_dir, f"data/counts/combined/output/combined-pc-counts-daily-stv2019.csv")
    
    if data_year == 2022:
        static_data_filepath         = os.path.join(data_dir, f"data/static/df1-2022.csv")
        monthly_strava_data_filepath = os.path.join(data_dir, f"data/counts/combined/output/combined-pc-counts-monthly-stv2022.csv")
        daily_strava_data_filepath   = os.path.join(data_dir, f"data/counts/combined/output/combined-pc-counts-daily-stv2022.csv")

    #
    #  Pick the right features based on <data_type> and <feature_set>
    #
    input_features = []
    input_features.extend(key_features)

    #
    #  Load the Static-Data first
    #
    static_df = pd.read_csv(static_data_filepath)
    print(f"Static Data Loaded with...{static_df.shape}")

    if "static" in feature_set:
        #  Include Input-Features
        input_features.extend(static_features)

    #
    #  Load the right Strava Data (Monthly)
    #
    if "stv-m" in feature_set:
        strava_df = pd.read_csv(monthly_strava_data_filepath)
        print(f"Monthly Strava Data Loaded with...{strava_df.shape}")

        #  Include Monthly-Strava Input-Features
        input_features.extend(monthly_features)
        input_features.extend(monthly_strava_features)

        #
        #  Join the Static-Data and Monthly-Strava-Data into Joint-Data
        if "static" in feature_set:
            df = pd.merge(strava_df, static_df, how='left', on='site_id')
            print(f"Joint Data Loaded with....{df.shape}")
        else:
            df = pd.merge(static_df, strava_df, how='left', on='site_id')
            print(f"Joint Data Loaded with....{df.shape}")

    #
    #  Load the right Strava Data (Daily)
    #
    if "stv-d" in feature_set:
        strava_df = pd.read_csv(daily_strava_data_filepath)
        print(f"Daily Strava Data Loaded with...{strava_df.shape}")

        #  Include Daily-Strava Input-Features
        input_features.extend(daily_features)
        input_features.extend(daily_strava_features)

        #
        #  Join the Static-Data and Daily-Strava-Data into Joint-Data
        if "static" in feature_set:
            df = pd.merge(strava_df, static_df, how='left', on='site_id')
            print(f"Joint Data Loaded with....{df.shape}")
        else:
            df = pd.merge(static_df, strava_df, how='left', on='site_id')
            print(f"Joint Data Loaded with....{df.shape}")

        #
        #  If we're using 'madb' as target, then we need monthly strava-data.
        if "madb" in target:
            strava_df = pd.read_csv(monthly_strava_data_filepath)
            df = pd.merge(df, strava_df[['site_id', 'month', 'madb']], how='left', on=['site_id', 'month'])
            print(f"Joint Data Loaded with....{df.shape}")


    #
    #  If it's just static data, finalize that with df.
    #
    if "stv-m" not in feature_set and \
       "stv-d" not in feature_set:
        df = static_df

    #
    #  Preprocess Data: Add previous target to each site_id/month pair.
    #
    if model_type == 'LSTM' and 'stv-m' in feature_set:
        prev_month_count = 1
        prev_targets = get_prev_months_target(df, target, prev_month_count)
        for i in range(prev_month_count):
            new_feature = f'prev_{target}[{i}]'
            df.insert(0, new_feature, prev_targets[i])
            input_features.append(new_feature)

    #
    #  Preprocess Data: From the daily-strava data, add the day and day-of-the-year columns
    #
    if model_type == 'LSTM' and 'stv-d' in feature_set or \
       model_type == 'DNN'  and 'stv-d' in feature_set:
        days, days_of_year = convert_date_to_day_of_year(df)

        new_feature = f'day'
        df.insert(0, new_feature, days)
        input_features.append(new_feature)

        new_feature = f'day_of_year'
        df.insert(0, new_feature, days_of_year)
        input_features.append(new_feature)
        
    #
    #  Preprocess Data: From the daily-strava data, add the previous day's target to the input-features
    #
    if model_type == 'LSTM' and 'stv-d' in feature_set:
        prev_day_count = 1
        prev_targets = get_prev_days_target(df, target, prev_day_count)
        for i in range(prev_day_count):
            new_feature = f'prev_{target}[{i}]'
            df.insert(0, new_feature, prev_targets[i])
            input_features.append(new_feature)

    #
    #  Remove site_id if needed
    #
    if "no-site-id" in feature_set:
        df = df.drop(columns=['site_id'])

        #  Exclude site_id from Input-Features
        input_features.remove('site_id')

    return df, input_features

#
#  Preprocess-Data
#
def preprocess_data(config_dict, df, input_features):
    target = config_dict['target']

    #
    #  Preprocess Data: Drop rows with NA values.
    #
    df = df.dropna()

    #
    #  Preprocess Data: Remove ' mph' suffix from maxspeed columns.
    #
    remove_mph = lambda s: s.removesuffix(' mph')
    if 'maxspeed_hm' in input_features:
        df.loc[:, 'maxspeed_hm'] = df['maxspeed_hm'].apply(remove_mph)
    if 'maxspeed_om' in input_features:
        df.loc[:, 'maxspeed_om'] = df['maxspeed_om'].apply(remove_mph)

    #
    #  Split Input-Data
    #
    x_df = df[input_features]

    #
    #  Split Output-Data
    #
    y_df = df[target]

    return x_df, y_df

def scale_data(x_df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(x_df)
    x_df = pd.DataFrame(scaled_data, columns=x_df.columns)

    return x_df, scaler

def reshape_data(config_dict, x_df):
    sample_count  = x_df.shape[0]
    feature_count = x_df.shape[1]
    if config_dict['model_type'] == 'LSTM':
        x_df = x_df.reshape((sample_count, 1, feature_count))
    if config_dict['model_type'] == 'DNN':
        x_df = x_df.reshape((sample_count, feature_count))

    return x_df

def split_data(x_df, y_df):
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.10, random_state=42)

    return (X_train, X_test, y_train, y_test)

#
#  Get Previous Month's Targets
#
def get_prev_months_target(df, target, prev_month_count):
    prev_targets = [[] for _ in range(prev_month_count)]

    samples = df.shape[0]
    for i in range(samples):
        site_id     = df.site_id[i]
        this_month  = df.month[i]

        for j in range(prev_month_count):
            prev_month  = this_month - j - 1

            prev_target = df[(df.site_id == site_id) & (df.month == prev_month)][target]

            if prev_target.empty is True:
                prev_targets[j].append(None)
            else:
                prev_targets[j].append(prev_target.item())

    return prev_targets

#
#  Get Previous Month's Targets
#
def get_prev_days_target(df, target, prev_day_count):
    prev_targets = [[] for _ in range(prev_day_count)]

    samples = df.shape[0]
    for i in range(samples):
        site_id          = df.site_id[i]
        this_day_of_year = df.day_of_year[i]

        for j in range(prev_day_count):
            prev_day_of_year = this_day_of_year - j - 1
            
            prev_target = df[(df.site_id == site_id) & (df.day_of_year == prev_day_of_year)][target]

            if prev_target.empty is True:
                prev_targets[j].append(None)
            else:
                prev_targets[j].append(prev_target.item())
    
    return prev_targets

def convert_date_to_day_of_year(df):
    days_of_year = []
    days = []

    samples = df.shape[0]
    for i in range(samples):
        date = df['Date'][i]

        dt = time.strptime(date, "%Y-%m-%d")

        day_of_year = dt.tm_yday
        day         = dt.tm_mday

        days_of_year.append(day_of_year)
        days.append(day)

    return days, days_of_year

#
#  Write-Data
#
def write_data(config_dir, config_dict, input_features, ds, df, scaler):
    data_year = config_dict['data_year']

    #
    #  Write config_dict to file
    #
    config_dict_filepath = os.path.join(config_dir, "config_dict.txt")
    with open(config_dict_filepath, 'w') as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}" + '\n')

    #
    #  Write df to file
    #
    for i, df_name in enumerate(['x_df', 'y_df']):
        data_filepath = os.path.join(config_dir, f"{data_year}-{df_name}.csv")
        with open(data_filepath, mode='w') as f:
            df[i].to_csv(f)

    #
    #  Write ds to file
    #
    for i, df_name in enumerate(['x_train', 'x_eval', 'y_train', 'y_eval']):
        data_filepath = os.path.join(config_dir, f"{data_year}-{df_name}.csv")
        with open(data_filepath, mode='w') as f:
            ds[i].to_csv(f)

    #
    #  Write input_features to file
    #
    input_features_df = pd.DataFrame(input_features)
    input_features_filepath = os.path.join(config_dir, f"{data_year}-input_features.csv")
    with open(input_features_filepath, mode='w') as f:
        input_features_df.to_csv(f)

    #
    #  Write scaler to file
    #
    scaler_filepath = os.path.join(config_dir, f"scaler_{data_year}.pkl")
    with open(scaler_filepath, 'wb') as f:
        pickle.dump(scaler, f)

    scaler_filepath = os.path.join(config_dir, f"scaler_{data_year}.csv")
    with open(scaler_filepath, 'w') as f:
        feature_count = scaler.n_features_in_
        feature_names = scaler.feature_names_in_
        scales = scaler.scale_
        mins = scaler.min_

        for column in ["feature_name", "scale", "min"]:
            f.write(f"{column}, ")
        f.write('\n')

        for i in range(feature_count):
            f.write(f"{feature_names[i]}, ")
            f.write(f"{scales[i]}, ")
            f.write(f"{mins[i]}, ")
            f.write(f"\n")

    return

#
#  Load, Preprocess, Scale, and Split the data.
#
def prepare_data(data_dir, config_dict):
    data, input_features = load_data(data_dir, config_dict)
    x_df, y_df = preprocess_data(config_dict, data, input_features)
    x_scaled_df, scaler = scale_data(x_df)
    dataset = split_data(x_scaled_df, y_df)

    return input_features, dataset, (x_df, y_df), scaler

#
#  Plot Functions
#
def plot_layers_vs_nodes(counts, lists, filepath, batch_size_idx):
    (layer_counts, node_counts, batch_sizes) = counts
    (I, J, K) = (len(node_counts), len(layer_counts), len(batch_sizes))
    (model_str_list, history_list) = lists

    fig, axs = plt.subplots(len(node_counts), len(layer_counts), figsize=(16, 16), sharex=True, sharey=True)
    fig.suptitle(f"Layer-Size vs Node-Count")
    for i, node_count in enumerate(node_counts):
        for j, layer_count in enumerate(layer_counts):
            k = batch_size_idx
            index = i*J*K + j*K + k

            model_str = model_str_list[index]
            loss = np.asarray(history_list[index]['loss'])
            val_loss = np.asarray(history_list[index]['val_loss'])

            axs[i, j].set_title(f"{model_str}", fontsize=12, pad=-1)
            axs[i, j].plot(loss, linewidth=1)
            axs[i, j].plot(val_loss, linewidth=1)

            val_loss_min_idx = np.argmin(val_loss)
            val_loss_min_val = np.min(val_loss)
            axs[i, j].plot(val_loss_min_idx, val_loss_min_val, '-ro')

            label=f"({val_loss_min_idx:03.0f}, {val_loss_min_val:02.0f})"
            axs[i, j].text(.01, .99, label, ha='left', va='top', transform=axs[i, j].transAxes, fontsize=12)

    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(filepath)
    plt.close(fig)

def plot_loss(history, loss_plot_filepath):
    fig, ax = plt.subplots()
    loss = history['loss']
    val_loss = history['val_loss']
    ax.plot(loss, label='loss')
    ax.plot(val_loss, label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(True)

    val_loss_min_idx = np.argmin(val_loss)
    val_loss_min_val = np.min(val_loss)

    ax.plot(val_loss_min_idx, val_loss_min_val, '-ro')
    label=f"({val_loss_min_idx:03.0f}, {val_loss_min_val:02.0f})"
    ax.annotate(label, (val_loss_min_idx, val_loss_min_val))

    fig.savefig(loss_plot_filepath)
    plt.close(fig)

#
#  Write Evaluation Results
#
def write_evaluation_results(config_dir, config_dict, results):
        data_year = config_dict['data_year']
        data_filepath = os.path.join(config_dir, f"Results-of-Evaluation-on-{data_year}-Data.txt")
        with open(data_filepath, mode='w') as f:
            for metric, value in results.items():
                f.write(f"{metric:<40s} = {value:>12.4f}" + '\n')

        data_filepath = os.path.join(config_dir, f"Results-of-Evaluation-on-{data_year}-Data.csv")
        with open(data_filepath, mode='w') as f:
            for key in results.keys():
                f.write(f"{key}, ")
            f.write('\n')

            for value in results.values():
                f.write(f"{value}, ")
            f.write('\n')

def write_predictions(config_dir, config_dict, df, predictions):
        x_df, y_df = df

        x_df.insert(0, 'true', y_df)
        x_df.insert(0, 'pred', predictions)
        
        data_year = config_dict['data_year']
        data_filepath = os.path.join(config_dir, f"Model-Predictions-on-{data_year}-Data.csv")
        with open(data_filepath, mode='w') as f:
            x_df.to_csv(f)

