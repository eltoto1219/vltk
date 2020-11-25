from dataclasses import dataclass, fields
import os


#okay so what is myh concern, I think that all that I need to worry about is importing this class for now
#okay so what are the steps, lets start top down
'''

6. finally in the config fucntion, we will init a class of the data extraction stuff and it will make our lives easier
7. once we get to that part though, we can include the model in the extraction class for now, but we can abstract away
later
8. last thing will be to chek the pixel mean stuff, and once that is done, we can set up the code to extract, lets try
to use 3 gpus for coco, and then we can use 1 gpu for visual_genome
'''

@dataclass
class ROIFeaturesFRCNN:

    out_file: str
    input_dir: str
    batch_size: int = 4
    log_name: str = 'extract_logs.txt'
    config_path: str = ''

    def __init__(self, out_file, input_dir, **kwargs):
        self.out_file = out_file
        self.input_dir = input_dir
        for field in fields(self):
            str_field = field.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))

@dataclass
class Environment:

    log_dir: str = (
            os.path.join(os.environ.get("HOME"), "logs")
            if os.environ.get("HOME", False) else os.path.join(os.getcwd(), "logs")
            )
    data_dir: str = '/playpen1/home/avmendoz/data'
    output_dir: str = '/playpen1/home/avmendoz/outputs'
    gpus: int = 1


    def __init__(self, **kwargs):
        for field in fields(self):
            str_field = field.name
            if str_field in kwargs:
                setattr(self, str_field, kwargs.get(str_field))

