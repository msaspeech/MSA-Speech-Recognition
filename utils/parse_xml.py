import xml.etree.ElementTree as ET
from .transcript import Transcript


def get_transcriptions(file_path):
    transcriptions = []

    tree = ET.parse(file_path)
    root = tree.getroot()
    for body in root.find("body"):

        for segment in body.findall("segment"):
            start_time = float(segment.attrib["starttime"])
            end_time = float(segment.attrib["endtime"])
            transcript = ""
            for element in segment.findall("element"):
                elt = element.text
                if elt is not None :
                    transcript += elt + " "
            transcriptions.append(Transcript(start_time, end_time, transcript))
            #transcriptions.append(transcript)
    return transcriptions


def generate_transcriptions_file(transcriptions_desc, output_path):
    transcriptions_text = []
    for t in transcriptions_desc :
        transcriptions_text.append(t.content)

    with open(output_path, "w") as output_file:
        output_file.writelines(transcriptions_text)
        output_file.close()
