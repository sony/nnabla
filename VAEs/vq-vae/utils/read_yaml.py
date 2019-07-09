import yaml

def read_yaml():
	with open("configs/config.yaml", 'r') as infile:
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