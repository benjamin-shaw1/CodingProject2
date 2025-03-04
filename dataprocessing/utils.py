import os 
import sys
from dotenv import load_dotenv, find_dotenv
import panel as pn

class utils:
    def __init__(self):
        pass
    def get_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("API_Key")
    
    def get_openapi_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("OPENAI_API_KEY")
    
    def get_api_url(self):
        _ = load_dotenv(find_dotenv())
        return os.getev("API_Url")

class upld_file():
    def __init__(self):
        self.widget_file_upload = pn.widgets.FileInput(accept = '.pdf,.ppt.png.html.pptx', multiple = False)
        self.widget_file_upload.param.watch(self.save_filename,"filename")
        
    def save_filename(self,_):
        if len(self.widget_file_upload.value) >2e6:
            print("file too large. 2 M limit")
        else:
            self.widget_file_upload.save('./example_files/'+self.widget_file_upload.filename)
            
        
        