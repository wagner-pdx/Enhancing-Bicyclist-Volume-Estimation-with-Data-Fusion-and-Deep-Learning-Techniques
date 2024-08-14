import os
import numpy as np

import Model_Configuration

# Filter out INFO and WARNING messages in tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from matplotlib import pyplot as plt

import data_utils
import exec_utils

months = 12

base_dir = "/home/wagner/"
data_dir = "/home/wagner/wsdot_techTransfer_apr22/"

def main():
    if base_dir is None:
        print(f"base_dir is None; it shouldn't be.")
        print(f"this is where we will put the Results dir.")
        return
    else:
        print(f"base_dir = {base_dir}")

    if data_dir is None:
        print(f"data_dir is None; it shouldn't be.")
        print(f"this is where we will get the data from.")
        return
    else:
        print(f"data_dir = {data_dir}")

    #
    #  Create the model_configurations
    #
    model_configurations = []
    DNN_monthly_configurations = Model_Configuration.create_DNN_monthly_configurations()
    model_configurations.extend(DNN_monthly_configurations)
    LSTM_monthly_configurations = Model_Configuration.create_LSTM_monthly_configurations()
    model_configurations.extend(LSTM_monthly_configurations)

    #
    #  Detail the Model_Configurations we have created.
    #
    print(f"model_configurations = {len(model_configurations)}")    
    for i, model_configuration in enumerate(model_configurations):
        model_str = model_configuration.get_str()
        print(f"    [{i+1:02.0f}]  {model_str}")

    #
    #  Draw monthly graphs for each configuration
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

        if 'no-site-id' in config_dict['feature_set']:
            continue

        results_dir, model_dir = create_dirs(base_dir, model_str)

        input_features, x_dfs, y_dfs, scalers = load_data_both(data_dir, config_dict)

        model = create_and_load_model(model_str, config_dict, input_features, model_dir)

        draw_monthly_target_prediction_by_site_id(model, model_dir, model_configuration, x_dfs, y_dfs, scalers)

    return

def create_dirs(base_dir, model_str):
    results_dir = os.path.join(base_dir, f"Results")
    os.makedirs(results_dir, exist_ok = True)

    model_dir = os.path.join(results_dir, model_str)
    os.makedirs(model_dir, exist_ok = True)

    return results_dir, model_dir

def load_data(data_dir, config_dict):
    data, input_features = data_utils.load_data(data_dir, config_dict)
    x_df, y_df = data_utils.preprocess_data(config_dict, data, input_features)
    _, scaler = data_utils.scale_data(x_df)

    return input_features, x_df, y_df, scaler

def load_data_both(data_dir, config_dict):
    config_dict_2019 = config_dict.copy()
    config_dict_2019['data_year'] = 2019
    input_features_2019, x_df_2019, y_df_2019, scaler_2019 = load_data(data_dir, config_dict_2019)

    config_dict_2022 = config_dict.copy()
    config_dict_2022['data_year'] = 2022
    input_features_2022, x_df_2022, y_df_2022, scaler_2022 = load_data(data_dir, config_dict_2022)

    if config_dict['data_year'] == 2019:
        input_features = input_features_2019

        x_dfs   = (x_df_2019, x_df_2022)
        y_dfs   = (y_df_2019, y_df_2022)
        scalers = (scaler_2019, scaler_2022)

    if config_dict['data_year'] == 2022:
        input_features = input_features_2022

        x_dfs   = (x_df_2022, x_df_2019)
        y_dfs   = (y_df_2022, y_df_2019)
        scalers = (scaler_2022, scaler_2019)

    return input_features, x_dfs, y_dfs, scalers

def create_and_load_model(model_str, config_dict, input_features, model_dir):
    model = exec_utils.create_model(model_str, config_dict, model_dir, input_features)

    history_filepath = os.path.join(model_dir, "history.txt")
    history = exec_utils.read_history(history_filepath)
    best_epoch, checkpoint_filepath = exec_utils.find_best_checkpoint(history, model_dir)
    if best_epoch is not None and checkpoint_filepath is not None:
        model.load_weights(checkpoint_filepath)

    return model

def get_dataset_loss_metrics(model, config_dict, x_dfs, y_dfs, scaler):

    if config_dict['model_type'] == 'DNN':
        metrics_0, pred_0 = exec_utils.evaluate_DNN_model(model, (x_dfs[0], y_dfs[0]), scaler)
        metrics_1, pred_1 = exec_utils.evaluate_DNN_model(model, (x_dfs[1], y_dfs[1]), scaler)

    if config_dict['model_type'] == 'LSTM':
        metrics_0, pred_0 = exec_utils.evaluate_LSTM_model(model, (x_dfs[0], y_dfs[0]), scaler)
        metrics_1, pred_1 = exec_utils.evaluate_LSTM_model(model, (x_dfs[1], y_dfs[1]), scaler)

    return (metrics_0, metrics_1)

def draw_monthly_target_prediction_by_site_id(model, model_dir, model_configuration, x_dfs, y_dfs, scalers):
    model_str   = model_configuration.get_str()
    config_dict = model_configuration.get_dict()
    h_lim = 5

    dataset_losses = get_dataset_loss_metrics(model, config_dict, x_dfs, y_dfs, scalers[0])

    data_year = config_dict['data_year']
    if data_year == 2019:
        data_years = (2019, 2022)
    else:
        data_years = (2022, 2019)

    fig, axs = plt.subplots(h_lim, 2, figsize=(8, 4), sharex=True, sharey='row', dpi=300)
    fig.tight_layout()

    metric = 'mean_absolute_error'
    fig.subplots_adjust(bottom=0.10, top=0.85, left=0.10, right=0.92)
    #fig.suptitle(f'{model_str} \n Predicted-Value vs True-Value \n {data_years[0]} data (L) and {data_years[1]} data (R) \n loss is {metric}', y = 0.99)
    fig.suptitle(f'{model_str} \n Predicted-Value vs True-Value', x = 0.55, y = 0.99)

    site_ids = list(set(x_dfs[0]['site_id']) & set(x_dfs[1]['site_id']))[:h_lim]
    for i, site_id in enumerate(site_ids):
        x_0, y_true_0, m_0 = filter_monthly_data_by_site_id(site_id, model, x_dfs[0], y_dfs[0], scalers[0])
        x_1, y_true_1, m_1 = filter_monthly_data_by_site_id(site_id, model, x_dfs[1], y_dfs[1], scalers[0])

        y_pred_0 = predict_monthly_target_by_site_id(model, config_dict, x_0, scalers[0])
        y_pred_1 = predict_monthly_target_by_site_id(model, config_dict, x_1, scalers[0])

        x_ticks = get_month_x_lims()
        y_ticks = get_month_y_lims(y_true_0, y_true_1, y_pred_0, y_pred_1)

        dataset_loss_0 = dataset_losses[0][metric]
        dataset_loss_1 = dataset_losses[1][metric]

        site_losses = get_site_loss_metrics(model, config_dict, (x_0, x_1), (y_true_0, y_true_1), scalers[0])
        site_loss_0 = site_losses[0][metric]
        site_loss_1 = site_losses[1][metric]

        #site_loss_0 = np.mean(np.abs(y_true_0 - y_pred_0.flatten()))
        #site_loss_1 = np.mean(np.abs(y_true_1 - y_pred_1.flatten()))

        make_monthly_by_site_id_subplot(axs[i, 0], m_0, y_true_0, y_pred_0, site_id, data_years[0], x_ticks, y_ticks, dataset_loss_0, site_loss_0, config_dict)
        make_monthly_by_site_id_subplot(axs[i, 1], m_1, y_true_1, y_pred_1, site_id, data_years[1], x_ticks, y_ticks, dataset_loss_1, site_loss_1, config_dict)

    lines, labels = axs[0, 0].get_legend_handles_labels()
    #fig.legend(lines, labels, loc = 'upper left', ncol=1, labelspacing=0., fontsize='large', mode='expand', borderpad=0.5)
    fig.legend(lines, labels, loc = 'upper left', ncol=2, fontsize=10, bbox_to_anchor=(0.03, 0.98))
    #fig.legend(lines, labels, loc = 'upper left', ncol=1, fontsize='large')

    filepath = os.path.join(model_dir, f"{model_str}_by-Site-ID_vs-other-year.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Wrote {filepath}")

    #plt.show()

    return

def filter_monthly_data_by_site_id(site_id, model, x_df, y_df, scaler):
    idx = (x_df.site_id == site_id).to_list()

    months = x_df.loc[idx, 'month'].values.tolist()

    x_np = x_df[idx]
    y_np = y_df[idx]

    return x_np, y_np, months

def predict_monthly_target_by_site_id(model, config_dict, x_np, scaler):
    model_type = config_dict['model_type']

    if model_type == 'DNN':
        x_scaled = scaler.transform(x_np)
        y_pred = model.predict(x_scaled)

    if model_type == 'LSTM':
        #x_np = x_np.to_numpy().reshape(x_np.shape[0], 1, x_np.shape[1])
        x_scaled = scaler.transform(x_np).reshape((x_np.shape[0], 1, x_np.shape[1]))
        y_pred = model.predict(x_scaled)

    return y_pred

def get_month_x_lims():
    x_len = months
    values = np.arange(0, x_len)+1
    labels = [ f"{x:.0f}" for x in values ]

    return {'values': values, 'labels': labels}

def get_month_y_lims(y_true_0, y_true_1, y_pred_0, y_pred_1):
    y_max = max(y_true_0.max(), y_true_1.max(), y_pred_0.max(), y_pred_1.max())
    y_max_lim = ((y_max // 1) + 1)

    y_min = min(y_true_0.min(), y_true_1.min(), y_pred_0.min(), y_pred_1.min())
    y_min_lim = ((y_min // 1) + 0)

    step = (y_max_lim - y_min_lim) / 2

    values = np.arange(y_min_lim, y_max_lim+1, step=step)

    labels = [ f"{y:.1f}" for y in values ]

    return {'values': values, 'labels': labels}

def get_site_loss_metrics(model, config_dict, x_dfs, y_dfs, scaler):

    if config_dict['model_type'] == 'DNN':
        metrics_0, preds_0 = exec_utils.evaluate_DNN_model(model, (x_dfs[0], y_dfs[0]), scaler)
        metrics_1, preds_1 = exec_utils.evaluate_DNN_model(model, (x_dfs[1], y_dfs[1]), scaler)

    if config_dict['model_type'] == 'LSTM':
        metrics_0, preds_0 = exec_utils.evaluate_LSTM_model(model, (x_dfs[0], y_dfs[0]), scaler)
        metrics_1, preds_1 = exec_utils.evaluate_LSTM_model(model, (x_dfs[1], y_dfs[1]), scaler)

    return (metrics_0, metrics_1)

def make_monthly_by_site_id_subplot(ax, x, y_true, y_pred, site_id, year, x_ticks, y_ticks, dataset_loss, loss, config_dict):
    y_true = y_true.to_numpy()

    ax.set_title(f"Site-ID = {site_id}, Year = {year}, Dataset-Loss = {dataset_loss:.1f}, Site-Loss = {loss:.1f}", y=0.95, fontsize=7)
    ax.plot(x, y_pred, '.-', linewidth=1, markersize=8, color='black', label=f"pred")
    ax.plot(x, y_true, '.-', linewidth=1, markersize=8, color='green', label=f"real")

    ax.grid()

    ax.set_xlabel(f"Months", fontsize=8)
    ax.set_xticks(x_ticks['values'])
    ax.set_xticklabels(x_ticks['labels'], fontsize=8)

    ax.set_ylabel(f"{config_dict['target']}", fontsize=8)
    ax.set_yticks(y_ticks['values'])
    ax.set_yticklabels(y_ticks['labels'], fontsize=8)

    ax.label_outer()

    #ax.text(.01, .99, label, ha='left', va='top', transform=ax.transAxes, fontsize=10)

    return

if __name__ == '__main__':
    main()
