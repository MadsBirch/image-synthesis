# -*- coding: utf-8 -*-
import click
import logging
import zipfile, os
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import shutil
import tarfile
import nibabel as nib
import numpy as np
from tqdm import tqdm
import random
import glob

logger = logging.getLogger(__name__)

def extract_brats_tar(tar_path, extract_path):
    
    logger.info(f'Extracting .tar file from {tar_path} to {extract_path}')
    
    file = tarfile.open(tar_path)
    file.extractall(extract_path)
    file.close()

def setup_directory_structure():
    
    logger.info("Setting up data folders.")

    if os.path.isdir('data'):
        shutil.rmtree('data')
        
    os.makedirs('data/processed')
    os.makedirs('data/raw')
    
    train_path = os.path.join('data/processed', "train")
    val_path = os.path.join('data/processed', "val")
    
    os.mkdir(train_path)
    os.mkdir(val_path)


def preprocess_brats(output_filepath:list, modalities:list):
    
    logger.info("Pre-processing MRI images.")
    cases_list = next(os.walk('data/raw'))[1]
    
    for case in tqdm(cases_list):
        
        case_path = os.path.join('data/raw',case)
        case_modality_list = []
        
        for modality in modalities:
            
            modality_mri_path = os.path.join(case_path, case + '_' + modality + '.nii.gz')
            modality_mri = nib.load(modality_mri_path)
            case_modality_list.append(modality_mri.get_fdata())
            
        # stack
        stacked_mri = np.stack(case_modality_list, axis = -1)
            
        # save nib
        sample_mri_image = nib.Nifti1Image(stacked_mri, affine=modality_mri.affine,
                                               header=modality_mri.header)
        sample_mri_path = os.path.join(output_filepath, case + '.nii.gz')
        nib.save(sample_mri_image, sample_mri_path)
        
        shutil.rmtree(case_path)
        

def train_val_split(output_filepath, data_split=0.8):
        
    samples = glob.glob(os.path.join(output_filepath, '*.nii.gz'))
    samples_train = random.sample(samples, int(len(samples)*data_split))
    samples_valid = [mri_sample for mri_sample in samples
                             if mri_sample not in samples_train]
    
    for sample in samples_train:
        new_sample_path = os.path.join(output_filepath, 'train', sample)
        os.rename(sample, new_sample_path)
    
    for sample in samples_valid:
        new_sample_path = os.path.join(output_filepath, 'val', sample)
        os.rename(sample, new_sample_path)
        

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    logger.info('making final data set from raw data')
    
    setup_directory_structure()
    
    # extract zip file
    extract_brats_tar(input_filepath, 'data/raw')
    
    # preprocess
    modalities = ['t1', 't2', 'flair']
    preprocess_brats(output_filepath, modalities)
    
    # train val split
    train_val_split(output_filepath, data_split=0.8)
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
