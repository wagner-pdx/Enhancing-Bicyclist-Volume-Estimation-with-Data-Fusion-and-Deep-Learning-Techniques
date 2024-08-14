import re
from string import Template

model_vars = \
    [
        'model_type',
        'data_year',
        'feature_set',
        'target',
        'node_count',
        'layer_count',
        'batch_size',
        'epochs',
    ]

def create_DNN_daily_configurations():
    model_types  = ['DNN']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static+stv-d',
            'static+stv-d+no-site-id',
            'stv-d',
            'stv-d+no-site-id',
        ]
    targets      = ['aadb', 'madb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [50]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                         data_years, 
                                                         feature_sets, 
                                                         targets, 
                                                         node_counts, 
                                                         layer_counts, 
                                                         batch_sizes,
                                                         epochs)
    
    return model_configurations

def create_DNN_monthly_configurations():
    model_types  = ['DNN']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static+stv-m',
            'static+stv-m+no-site-id',
            'stv-m',
            'stv-m+no-site-id',
        ]
    targets      = ['aadb', 'madb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [1000]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                           data_years, 
                                                           feature_sets, 
                                                           targets, 
                                                           node_counts, 
                                                           layer_counts, 
                                                           batch_sizes,
                                                           epochs)
    
    return model_configurations

def create_DNN_yearly_configurations():
    model_types  = ['DNN']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static',             # Static-Only feature-set and MADB target are mutually exclusive.
            'static+no-site-id',  # Static-Only feature-set and MADB target are mutually exclusive.
        ]
    targets      = ['aadb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [1000]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                         data_years, 
                                                         feature_sets, 
                                                         targets, 
                                                         node_counts, 
                                                         layer_counts, 
                                                         batch_sizes,
                                                         epochs)
    
    return model_configurations

def create_LSTM_daily_configurations():
    model_types  = ['LSTM']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static+stv-d',
            'static+stv-d+no-site-id',
            'stv-d',
            'stv-d+no-site-id',
        ]
    targets      = ['aadb', 'madb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [50]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                         data_years, 
                                                         feature_sets, 
                                                         targets, 
                                                         node_counts, 
                                                         layer_counts, 
                                                         batch_sizes,
                                                         epochs)
    
    return model_configurations

def create_LSTM_monthly_configurations():
    model_types  = ['LSTM']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static+stv-m',
            'static+stv-m+no-site-id',
            'stv-m',
            'stv-m+no-site-id',
        ]
    targets      = ['aadb', 'madb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [1000]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                         data_years, 
                                                         feature_sets, 
                                                         targets, 
                                                         node_counts, 
                                                         layer_counts, 
                                                         batch_sizes,
                                                         epochs)
    
    return model_configurations

def create_LSTM_yearly_configurations():
    model_types  = ['LSTM']
    data_years   = [2019, 2022]
    feature_sets = \
        [
            'static',             # Static-Only feature-set and MADB target are mutually exclusive.
            'static+no-site-id',  # Static-Only feature-set and MADB target are mutually exclusive.
        ]
    targets      = ['aadb']
    node_counts  = [512]
    layer_counts = [2]
    batch_sizes  = [8]
    epochs       = [1000]

    model_configurations = Generate_Model_Configurations(model_types, 
                                                         data_years, 
                                                         feature_sets, 
                                                         targets, 
                                                         node_counts, 
                                                         layer_counts, 
                                                         batch_sizes,
                                                         epochs)
    
    return model_configurations


def Generate_Model_Configurations(model_types, 
                                  data_years, 
                                  feature_sets, 
                                  targets, 
                                  node_counts, 
                                  layer_counts, 
                                  batch_sizes, 
                                  epochses):
    model_configurations = []

    for model_type in model_types:
        for data_year in data_years:
            for feature_set in feature_sets:
                for target in targets:
                    for node_count in node_counts:
                        for layer_count in layer_counts:
                            for batch_size in batch_sizes:
                                for epochs in epochses:
                                    config_dict = \
                                    {
                                        'model_type'  : model_type,
                                        'data_year'   : data_year,
                                        'feature_set' : feature_set,
                                        'target'      : target,
                                        'node_count'  : node_count,
                                        'layer_count' : layer_count,
                                        'batch_size'  : batch_size,
                                        'epochs'      : epochs,
                                    }
                                    model_configuration = Model_Configuration(_config_dict=config_dict)
                                    model_configurations.append(model_configuration)

    return model_configurations

class Model_Configuration():
    def __init__(self, 
                 _model_str = None, 
                 _config_dict = None):
        self.re = re.compile(r"^([^(]*)\(([^)]*)\)\(([^)]*)\)\(([^)]*)\)\(([^)]*)\)_([0-9]*)x([0-9]*)_Batch-Size-([0-9]*)")
        self.str_template = Template("${model_type}(${data_year})(${feature_set})(${target})_${node_count}x${layer_count}_Batch-Size-${batch_size}")

        if _config_dict is not None:
            self.config_dict = _config_dict
            model_str = self.dict_to_str()

            if model_str is not None:
                self.model_str = model_str
            else:
                print("Given bad config-dictionary to create Model_Configuration")
                missing_keys = self.get_missing_keys()
                print(f"    missing keys : {missing_keys}")

        if _model_str is not None:
            self.model_str = _model_str
            config_dict = self.str_to_dict()

            if config_dict is not None:
                self.config_dict = config_dict
            else:
                print("Given bad model-str to create Model_Configuration")

    def verify_dict(self):
        key_complete = self.verify_dict_keys()

        if key_complete is True:
            complete = self.verify_dict_values()
        else:
            complete = False
        
        return complete

    def verify_dict_keys(self):
        complete = True

        for model_var in model_vars:
            if model_var not in self.config_dict.keys():
                complete = False

        return complete
    
    def verify_dict_values(self):
        complete = True

        for model_var in model_vars:
            if self.config_dict[model_var] is None:
                complete = False

        return complete

    def get_missing_keys(self):
        missing_keys = []

        for model_var in model_vars:
            if model_var not in self.config_dict.keys():
                missing_keys.append(model_var)

        return missing_keys

    def str_to_dict(self):
        re_match = self.re.fullmatch(self.model_str)

        if re_match is not None:
            self.model_type  = re_match.group(1)
            self.data_year   = re_match.group(2)
            self.feature_set = re_match.group(3)
            self.target      = re_match.group(4)
            self.node_count  = re_match.group(5)
            self.layer_count = re_match.group(6)
            self.batch_size  = re_match.group(7)
            self.epochs      = re_match.group(8)

            config_dict = \
            {
                'model_type'  : self.model_type,
                'data_year'   : self.data_year,
                'feature_set' : self.feature_set,
                'target'      : self.target,
                'node_count'  : self.node_count,
                'layer_count' : self.layer_count,
                'batch_size'  : self.batch_size,
                'epochs'      : self.epochs,
            }
        else:
            #print(f"the model_str provided didn't match the regulare expression I crafted for it.")
            #print(f"maybe check your model_str?")
            #print(f"maybe check the regular expression?")
            config_dict = None

        return config_dict

    def dict_to_str(self):
        dict_complete = self.verify_dict()

        if dict_complete is True:
            model_str = self.str_template.substitute(self.config_dict)
        else:
            model_str = None

        return model_str

    def get_dict(self):
        return self.config_dict
    
    def get_str(self):
        return self.model_str
