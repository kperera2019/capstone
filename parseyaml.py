
import yaml

my_dict = yaml.safe_load(open("toy_config//toy_config.yaml"))
print(my_dict.get('embeddings'))