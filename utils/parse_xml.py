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
            transcrpt = ""
            for element in segment.findall("element"):
                elt = element.text
                if elt is not None :
                    transcrpt += elt + " "
            transcriptions.append(Transcript(start_time, end_time, transcrpt))

    return transcriptions


def generate_transcriptions_file(input_path, output_path):
    transcriptions = get_transcriptions(file_path=input_path)
    with open(output_path, "w") as output_file:
        output_file.writelines(transcriptions)
        output_file.close()
