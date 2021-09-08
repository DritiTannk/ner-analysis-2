import os
import glob

from tika import parser


def pdf_to_text_using_tikka(filepath, filename, output_path):
    """
    This method extract text from PDF using tika library.
    """
    output_file_name = filename + '_tikka.txt'
    output_file_path = output_path + output_file_name
    parsed_file = parser.from_file(filepath)  # Parsing PDF file
    content = parsed_file['content']  # Extracting content

    with open(output_file_path, 'w') as f:
        res = f.write(content.strip())

    return res


def file_search(input_path, output_path):
    """
    This method gets files from given input path & checks extension.

    :param input_path: input files path
    :param output_path: output files path

    :return: statistics dictionary
    """
    pdf_counter = 0
    input_path = input_path + '*'

    for file in glob.glob(input_path):
        fname = os.path.basename(file)
        filename, ext = os.path.splitext(fname)

        if fname.endswith('.pdf') and ext == '.pdf':
            res = pdf_to_text_using_tikka(file, filename, output_path)

            if res > 0:
                pdf_counter += 1

    return pdf_counter


if __name__ == '__main__':
    i_path = input('Enter The Input Path: ')  # Get user's input files directory path
    o_path = input('Enter The Output Path: ')  # Get user's output files directory path

    i_path_exist = os.path.exists(i_path)  # Check the user input path
    o_path_exist = os.path.exists(o_path)  # Check the user output path

    if i_path_exist and o_path_exist:
        stats = file_search(i_path, o_path)

        print(f"\n Extraction Result \n {'-' * 20}"
              f"\n Total PDF Files ==> {stats} "
              )
    else:
        print("\n \n Given paths does not exists")
