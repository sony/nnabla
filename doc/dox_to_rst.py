import os
import sys
import xml.etree.ElementTree as ET

def transfer_filename(name):
    illegal_char = ["\\", "/", "?", "*", "$", "!", "&", "\'", "\"", "|", ";", "."]
    legal_name = ""
    for char in name:
        if char == ":":
            legal_name += "-"
        elif char == "<":
            legal_name += "("
        elif char == ">":
            legal_name += ")"
        elif char == " ":
            continue
        elif char in illegal_char:
            continue
        else:
            legal_name += char
    return legal_name

def write_rst(kind, name, project, path):
    file_name = transfer_filename(name)
    with open(os.path.join(path, 'cpp', project, kind + '.rst'), "a") as rst:
        rst.write("\t" + kind + "/" + file_name + ".rst\n")
        rst.close()
    with open(os.path.join(path, 'cpp', project, kind, file_name + '.rst'), "w") as subrst:
        subrst.write(kind + " " + name + "\n")
        subrst.write("========================================")
        subrst.write("\n\n.. doxygen" + kind + ":: " + name +"\n")
        subrst.write("\t:project: " + project +"\n\t:members: \n")
        subrst.close()

def check_not_empty(tree):
    root = tree.getroot()
    for comment in root.iter("briefdescription"):
        if comment.find("para") is not None:
            return True
    for comment in root.iter("detaileddescription"):
        if comment.find("para") is not None:
            return True
    return False

def parse_dox(project, path):

    tree = ET.parse(os.path.realpath(os.path.join(path, '..', 'build-doc', 'doc', 'xml-' + project, 'index.xml')))
    root = tree.getroot()
    toctree = ".. toctree::\n\tmaxdepth: 1\n\n"
    os.mkdir(os.path.join(path, 'cpp', project))
    os.mkdir(os.path.join(path, 'cpp', project, 'class'))
    os.mkdir(os.path.join(path, 'cpp', project, 'struct'))
    os.mkdir(os.path.join(path, 'cpp', project, 'namespace'))
    title = project.replace('Cpp', '').replace('-', ' ')
    with open(os.path.join(path, 'cpp', project, 'class.rst'), "w") as rst:
        rst.write("NNABLA " + title + "Class\n============\n\n" + toctree)
        rst.close()
    with open(os.path.join(path, 'cpp', project, 'struct.rst'), "w") as rst:
        rst.write("NNABLA " + title + "Struct\n=============\n\n" + toctree)
        rst.close()
    with open(os.path.join(path, 'cpp', project, 'namespace.rst'), "w") as rst:
        rst.write("NNABLA " + title + "Namespace\n================\n\n" + toctree)
        rst.close()

    for child in root:
        name = child.find("name").text
        if child.attrib["kind"] == "class":
            ref = child.attrib["refid"]
            subtree = ET.parse(os.path.realpath(os.path.join(path, '..', 'build-doc', 'doc', 'xml-' + project, ref + '.xml')))
            if check_not_empty(subtree):
                write_rst("class", name, project, path)
        elif child.attrib["kind"] == "struct": 
            ref = child.attrib["refid"]
            subtree = ET.parse(os.path.realpath(os.path.join(path, '..', 'build-doc', 'doc', 'xml-' + project, ref + '.xml')))
            if check_not_empty(subtree):
                write_rst("struct", name, project, path)
        elif child.attrib["kind"] == "namespace":
            ref = child.attrib["refid"]
            subtree = ET.parse(os.path.realpath(os.path.join(path, '..', 'build-doc', 'doc', 'xml-' + project, ref + '.xml')))
            if check_not_empty(subtree):
                write_rst("namespace", name, project, path)      
        else: continue

Cuda_opt = sys.argv[1]
current_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_path, 'cpp', 'doc.rst'), "w") as rst:
    rst.write("C++ API Document\n================\n\n")
    rst.write(".. toctree:: \n\tmaxdepth: 2\n\n")
    rst.write("\tcpp_api.rst\n")
    if Cuda_opt == '1':
        rst.write("\text_cuda_cpp_api.rst\n")
parse_dox('Cpp', current_path)
if Cuda_opt == '1':
    parse_dox('Ext-Cuda-Cpp', current_path)
