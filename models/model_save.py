from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials

def google_auth():
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    return drive

def save_to_drive(model_name, model_path):
    drive = google_auth()
    model_title = model_name.split(".h5")[0]
    uploaded = drive.CreateFile({model_title: model_name})
    uploaded.SetContentFile(model_path)
    uploaded.Upload()
    #print('Uploaded file with ID {}'.format(uploaded.get()))