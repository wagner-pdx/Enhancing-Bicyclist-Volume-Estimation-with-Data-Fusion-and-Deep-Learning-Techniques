
import Model_Configuration
import exec_utils

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

    model_configurations = []

    DNN_daily_configurations = Model_Configuration.create_DNN_daily_configurations()
    model_configurations.extend(DNN_daily_configurations)
    DNN_monthly_configurations = Model_Configuration.create_DNN_monthly_configurations()
    model_configurations.extend(DNN_monthly_configurations)
    DNN_yearly_configurations = Model_Configuration.create_DNN_yearly_configurations()
    model_configurations.extend(DNN_yearly_configurations)

    LSTM_daily_configurations = Model_Configuration.create_LSTM_daily_configurations()
    model_configurations.extend(LSTM_daily_configurations)
    LSTM_monthly_configurations = Model_Configuration.create_LSTM_monthly_configurations()
    model_configurations.extend(LSTM_monthly_configurations)
    LSTM_yearly_configurations = Model_Configuration.create_LSTM_yearly_configurations()
    model_configurations.extend(LSTM_yearly_configurations)
    
    #
    #  Detail the Model_Configurations we have created.
    #
    print(f"model_configurations = {len(model_configurations)}")    
    for i, model_configuration in enumerate(model_configurations):
        model_str = model_configuration.get_str()
        print(f"    [{i+1:02.0f}]  {model_str}")

    #
    #  Evaluate the configurations we created (and previously ran)
    #
    exec_utils.evaluate_all_configurations(base_dir,
                                           data_dir,
                                           model_configurations)

    return


if __name__ == '__main__':
    main()