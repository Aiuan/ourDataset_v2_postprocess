import os
from TIRadar.adcdata_decoder import RecordData_NormalMode, RecordData_MixMode

def main():
    root = 'F:\\20221217\\TIRadar'
    output_root = 'F:\\20221217_process\\TIRadar'
    for item in os.listdir(root):
        if 'mixmode' in item:
            record_data = RecordData_MixMode(os.path.join(root, item))
        else:
            record_data = RecordData_NormalMode(os.path.join(root, item))
        output_folder = os.path.join(output_root, record_data.folder)
        record_data.divide_frame(output_folder)

    print('done')

if __name__ == '__main__':
    main()
