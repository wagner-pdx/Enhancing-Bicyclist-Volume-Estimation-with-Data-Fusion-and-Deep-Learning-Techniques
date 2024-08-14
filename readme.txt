Q. What are the dependencies?
A. Tensorflow, numpy, pandas, matplotlib, sklearn
	I used VS-Code to develop, debug and often execution too.
	I used WSL-2/Ubuntu-24.04 on Windows 11

Q. What datafiles do I need to run this?
A. These are the files, put them in whatever you specify as <data_dir>
	data/static/df1-2019.csv
	data/counts/combined/output/combined-pc-counts-monthly-stv2019.csv
	data/counts/combined/output/combined-pc-counts-daily-stv2019.csv
	data/static/df1-2022.csv
	data/counts/combined/output/combined-pc-counts-monthly-stv2022.csv
	data/counts/combined/output/combined-pc-counts-daily-stv2022.csv


Q. What even are all these files?
	data_utils.py.......................There are the utilities/functions of or relating to...
											loading, preprocessing, scaling, spliting, synthesizing the data itself before execution.
	                 						there are also utilities about writing the loss-plots, the layers-vs-nodes plots, and writing evaluation/prediction results
	                 						however, the code for making the predicted/actual target vs month per site plots are not in here (but maybe should be)

	exec_utils.py.......................These are the utilities/functions of or relating to creating, training, evaluating the models
					 						there are also a few regarding reading/writing/merging 'histories' which are the records of training.
					 						Also for reading/writing/calculating execution duration
					 						Also Model-Checkpoint management (needs further testing)

	draw_data.py........................This is a whole executable that does the predicted/actual target vs month per site plots, 
					     					it actually loads up the model from the best checkpoint (according to history.txt)
					     					loads up the data, and executes the model to get the predictions,
					     					it goes on to make the plots
					     					(it certainly has redundant code found in the previous files, and could be consilidated later)

	feature_sets.py.....................This is essentially just a header where I conveniently define the feature sets.

	Model_configuration.py..............This is an object that I made and used to manage all the different configurations.
											the idea was that I give it arrays for each variable under consideration,
												ex) 3 values for number-of-layers, 3 values for number-of-nodes-per-layer, 3 values for batch-size, etc...
											and it'll go through and create a config every combination of those variables
												ex) 3x3x3 ... 27 difference configurations
											in this case, a config consists of a config-string and config-dict
											    the config-string is for human readability, and also for folder-names/storage, etc
											    the config-dict is where the real configuration is that gets loaded up for setup/execution
											we are given these in a list/array so that we can iterate through them and do what we need
												whether that be execution or evaluation

	Execute_ALL_configurations.py.......This should execute all the model training and evaluation for all the configurations.

	Evaluation_ALL_configurations.py....This should just evaluate all the configurations, assuming they're all trained and present and have...
											1. their best checkpoint present
											2. the history.txt from training (to be able to know what the best checkpoint IS)
											3. duration.txt, not strictly necessary, but still only generated/computed during execution/training

Q. Where will the results go?
A. They will go into <base_dir>/Results

Q. How do I run this?
A.  To run all the configurations, 
    use '$python Execute_ALL-configurations'

    To execute evaluation on the configurations that have been previously ran,
    use '$python Evaluation_ALL_configurations.py'