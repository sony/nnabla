import yaml
import os 

def read_yaml(data_config_file):
	config_file_path = os.path.join('configs', data_config_file)
	with open(config_file_path, 'r') as infile:
		try:
			return yaml.load(infile, Loader = yaml.FullLoader)
		except yaml.YAMLError as exc:
			return exc

def write_yaml(config):

	with open('configs/config.yaml', 'w') as outfile:
		try:
			yaml.dump(config, outfile, default_flow_style = False)
		except:
			print("Unable to update the config file")