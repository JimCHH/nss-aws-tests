import sys, glob, re
import pandas as pd

def correct(cases):
    for date_site_patient in cases:
        source = f'/home/ubuntu/S3/{date_site_patient.split("_")[1]}/Result/{date_site_patient}'
        for filepath in glob.glob(f'{source}/*.csv'):
            filepath_tsdf = filepath[:-4] + '_TimeSeriesDataFrame.csv'
            filepath_corr = filepath[:-4] + '_CORRECTED.csv'

            # Backward compatibility
            with open(filepath) as f:
                delimiter = f.readline()[4]
            
            if delimiter == ',':
                # Make CORRECTED.csv
                with open(filepath) as f, open(filepath_corr, 'w') as fcorr:
                    lines = f.readlines()
                    fcorr.write(''.join(lines[:5]))
                    for line in lines[5:]:
                        fcorr.write(re.sub(',(\d{6})', r'.\1', line))
                # Make TimeSeriesDataFrame.csv
                df = pd.read_csv(filepath_corr, skiprows=4)
                df.to_csv(filepath_tsdf, index=False)
            
            if delimiter == ';':
                # Make TimeSeriesDataFrame.csv
                df = pd.read_csv(filepath, skiprows=4, sep=';')
                if type(df.loc[0][1]) == str:
                    df = pd.read_csv(filepath, skiprows=4, sep=';', decimal=',')
                df.to_csv(filepath_tsdf, index=False)
                # Make CORRECTED.csv
                with open(filepath) as f, open(filepath_tsdf) as ftsdf, open(filepath_corr, 'w') as fcorr:
                    fcorr.write(''.join(f.readlines()[:4]).replace(';', ','))
                    fcorr.write(''.join(ftsdf.readlines()))

if __name__ == '__main__':
    correct(sys.argv[1:])