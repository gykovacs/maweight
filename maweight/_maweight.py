"""
Created on Wed Feb 14 00:34:03 2018

@author: gykovacs
"""

__all__= ['executable_version',
          'parameters_default',
          'parameters_fast',
          'register_and_transform',
          'extract_features_3d',
          'model_selection']

# to call OS services
import sys
# to work with filenames
import os.path
# to call executables
import subprocess
# for numerical methods
import numpy as np
# for data structures
import pandas as pd
# for niftii and other medical file formats
import nibabel as nib
# to create temporary files
import tempfile
# for simple descriptive statistics
from scipy.stats import skew, kurtosis
# to copy the resulting files
import shutil
# to read/write 2d images
import imageio

# for model selection
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from maweight.mltoolkit.automl import R2_score, RMSE_score
from maweight.mltoolkit.optimization import SimulatedAnnealing, UniformIntegerParameter, ParameterSpace, BinaryVectorParameter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

from maweight.mltoolkit.automl import *

def _find_executable(name):
    """ Try to find a executables by name.
    Args:
        name(str): name of the executable
    Returns:
        str: absolute path of the executable
    """
    exe_name = name + '.exe' * sys.platform.startswith('win')
    env_path = os.environ.get(name.upper()+ '_PATH', '')
    
    # prepare the list of candidate locations
    location_candidates= os.environ['PATH'].split(';' if sys.platform.startswith('win') else ':')
    location_candidates.extend([env_path,
                                  os.path.dirname(env_path),
                                  os.path.dirname(sys.executable),
                                  os.path.expanduser('~'),
                                  'c:\\program files', 
                                  os.environ.get('PROGRAMFILES'),
                                  'c:\\program files (x86)',
                                  os.environ.get('PROGRAMFILES(x86'),
                                  '/usr/bin',
                                  '/usr/local/bin',
                                  '/opt/local/bin',
                                  os.path.dirname(os.path.abspath(__file__ if '__file__' in locals() else '')),
                                  os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else ''))])
    
    location_candidates= [l for l in location_candidates if l and os.path.isdir(l)]
    location_candidates= list(set(location_candidates))
    
    # checking the candidate directories
    for d in location_candidates:
        file= os.path.join(d, exe_name)
        if os.path.isfile(file):
            try:
                subprocess.check_output([file])
                return file
            except:
                pass

    # checking subdirectories
    for d in location_candidates:
        for subd in sorted(os.listdir(d)):
            file= os.path.join(d, subd, exe_name)
            if os.path.isfile(file):
                try:
                    subprocess.check_output([file])
                    return file
                except:
                    pass
    
    return None

def executable_version():
    out, err= subprocess.Popen([_maweight_executables['elastix'], '--version'], stdout=subprocess.PIPE).communicate()
    return out

def parameters_default(image_dim= 3, default_pixel_value= 0):
    """
    Returns a default, general settings of the parameters.
    Args:
        image_dim (int or image): the number of image dimensions or
                                    an image to derive the number of
                                    dimensions from
        default_pixel_value (num): the default value of pixels
    Returns:
        dict: a dictionary of parameters
    """
    
    if not isinstance(image_dim, int):
        try:
            image_dim= len(image_dim.shape)
        except:
            raise ValueError("pass an integer or an image as 'image_dim'")
    
    params= {'FixedImageDimension': image_dim,
             'MovingImageDimension': image_dim,
             'WriteResultImage': "true",
             'ResultImagePixelType': "double",
             'FixedInternalImagePixelType': "float",
             'MovingInternalImagePixelType': "float",
             'UseDirectionCosines': "true",
             'Registration': "MultiResolutionRegistration",
             'FixedImagePyramid': "FixedRecursiveImagePyramid",
             'MovingImagePyramid': "MovingRecursiveImagePyramid",
             'HowToCombineTransforms': "Compose",
             'DefaultPixelValue': "%f" % default_pixel_value,
             'Interpolator': "BSplineInterpolator",
             'BSplineInterpolationOrder': "1",
             'ResampleInterpolator': "FinalBSplineInterpolator",
             'FinalBSplineInterpolationOrder': 3,
             'Resampler': "DefaultResampler",
             'Metric': "AdvancedMattesMutualInformation",
             'NumberOfHistogramBins': 32,
             'ImageSampler': "RandomCoordinate",
             'NumberOfSpatialSamples': 4048,
             'NewSamplesEveryIteration': "true",
             'NumberOfResolutions': 4,
             'Transform': "BSplineTransform",
             'Optimizer': "AdaptiveStochasticGradientDescent",
             'MaximumNumberOfIterations': 200,
             'FinalGridSpacingInVoxels': [10, 10, 10],
             'ResultImageFormat': "nii.gz",
             'CheckNumberOfSamples': "false",
             "RandomSeed": 5}
    
    return params
        
def parameters_fast(image_dim= 3, default_pixel_value= -1024):
    """
    Returns a faster settings of the parameters.
    Args:
        image_dim (int or image): the number of image dimensions or
                                    an image to derive the number of
                                    dimensions from
        default_pixel_value (num): the default value of pixels
    Returns:
        dict: a dictionary of parameters
    """
    params= parameters_default(image_dim, default_pixel_value)
    params['NumberOfSpatialSamples']= 1024
    params['NumberOfHistogramBins']= 32
    params['RandomSeed']= 5
    
    return params
    
def _save_parameters_to_file(params, filename):
    """
    Save parameters to file in the appropriate format.
    Args:
        params (dict): dictionary of parameters
        filename (str): the name of the file
    """
    
    file= open(filename, "w")
    for k in params:
        if isinstance(params[k], str):
            file.write('(%s "%s")\n' % (k, params[k]))
        elif isinstance(params[k], int):
            file.write('(%s %d)\n' % (k, params[k]))
        elif isinstance(params[k], float):
            file.write('(%s %f)\n' % (k, params[k]))
        elif isinstance(params[k], list):
            format_string= '(%s' + (' %d'*len(params[k])) + ')\n'
            file.write(format_string % tuple([k] + params[k]))
    file.close()

def _prepare_files(image_parameters, tmp_dir):
    """
    Those image parameters which are not file paths are saved as files in the
    temporary directory, the dimension of images if extracted.
    Args:
        image_parameters (list): the list of image parameters (paths or images)
        tmp_dir (str): the path of the temporary directory
    Returns:
        list, int: the list of paths, dimension of images
    """
    file_arguments= []
    image_dim= None
    
    for i, f in zip(range(len(image_parameters)), image_parameters):
        if isinstance(f, str):
            # if argument is filename
            file_arguments.append(f)
            if f.endswith('.nii') or f.endswith('nii.gz'):
                image_dim= 3
            elif f[-3:].lower() in ['tif']:
                image_dim= 2
        elif (isinstance(f, np.ndarray) and len(f.shape) == 3) or isinstance(f, nib.Nifti1Image) or isinstance(f, nib.Nifti2Image):
            # if argument is 3D array or NiftiImage object
            if isinstance(f, np.ndarray):
                f= nib.Nifti1Image(f, affine=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
            tmp_file= os.path.join(tmp_dir, str(i) + '.nii.gz')
            nib.save(f, tmp_file)
            file_arguments.append(tmp_file)
            image_dim= 3
        elif (isinstance(f, np.array) and len(f.shape) == 2):
            # if argument is 2D array
            tmp_file= os.path.join(tmp_dir, str(i) + '.tif')
            imageio.imwrite(tmp_file, f)
            file_arguments.append(tmp_file)
            image_dim= 2
    
    return file_arguments, image_dim

def _execute_command(command, verbose):
    """
    Executes a shell command and depending on the verbosity level prints its
    output to the stdout.
    Args:
        command (list(str)): the command and its arguments
        verbose (int): level of verbosity (0, 1, 2)
    """
    p= subprocess.Popen(command, stdin=subprocess.PIPE, stdout= subprocess.PIPE, stderr= subprocess.PIPE)
    if verbose:
        while p.poll() is None:
            out= p.stdout.readline()
            sys.stdout.write(out)
            sys.stdout.flush()
    else:
        p.communicate()

def register_and_transform(moving, 
                           fixed, 
                           to_fit, 
                           output_names= None, 
                           params= None, 
                           work_dir= None, 
                           threads= 6, 
                           elastix_path= None, 
                           transformix_path= None,
                           registered_image_path= None,
                           verbose= 1):
    """
    Registers and transforms images.
    Args:
        moving (str or ndarray or Nifti1Image): the moving image
        fixed (str or ndarray or Nifti1Image): the fixed image
        to_fit (list(str or ndarray or Nifti1Image)): the images to transform
                                with the deformation field extracted during the
                                registration
        output_names (list(str)): optional filenames to save the transformed images on
        work_dir (str): optional temporary directory path
        threads (int): the number of threads to use
        elastix_path (str): path of the elastix executable
        transformix_path (str): path of the transformix executable
        registered_image_path (str): path to save the registered image to
        verbose (int): verbosity level (0, 1, 2)
    Returns:
        list(str or Nifti1Image): output_names if provided, otherwise
                                the list of transformed Nifti1Image objects
                                
    Usage:
        Suppose you want to segment image F (fixed) using the segmented image M
        (moving) and two masks MA and MB. F, M, MA and MB can be either 
        numpy.ndarray, nibabel.Nifti1Image, or string paths to nii images.
        
        Calling the register_and_transform function will register M to F, and
        using the deformation field transforms MA and MB to F. (Arbitrary number
        of mask images can be used.)
        
        Example calls:
            1) The following call returns the two transformed masks in a list:
            
            register_and_transform(M, F, [MA, MB])
            
            2) The following call writes the two fitted masks into files
                path_a and path_b:
                    
            register_and_transform(M, F, [MA, MB], 
            output_names= [path_a, path_b])
            
            3) The following path uses a specified working directory:
    
            register_and_transform(M, F, [MA, MB],
                                   output_names= [path_a, path_b],
                                   work_dir= path_work_dir)
            
            4) Parameters of the registration can be altered by generating
                a default set of parameters calling the function
                
                def_par= parameters_default()
                
                updating the dict def_par arbitrarily, like
                
                def_par['NumberOfSpatialSamples']= 1024
                
                and passing it to the register_and_transform function:
                    
            register_and_transform(M, F, [MA, MB],
                                   output_names= [path_a, path_b],
                                   params= def_par)
            
            5) A particular example:
                
            register_and_transform('/home/gykovacs/rabbit/etalon/201k.nii.gz',
                                     '/home/gykovacs/workspaces/rabbit/to_segment/001-a.nii.gz',
                                     ['/home/gykovacs/workspaces/rabbit/etalon/201k-mld.nii.gz',
                                      '/home/gykovacs/workspaces/rabbit/etalon/201k-hinds.nii.gz'],
                                     ['/home/gykovacs/workspaces/rabbit/output/001-a-mld.nii.gz',
                                      '/home/gykovacs/workspaces/rabbit/output/001-a-hinds.nii.gz'],
                                     work_dir= '/home/gykovacs/tmp/')

    """
    
    if (not output_names is None) and len(output_names) != len(to_fit):
        raise Exception("length of output_names (%d) and to_fit (%d) do not match" % (len(output_names), len(to_fit)))
    
    elastix_path= elastix_path or _maweight_executables['elastix']
    transformix_path= transformix_path or _maweight_executables['transformix']
    
    if elastix_path is None or transformix_path is None:
        raise Exception('elastix and transformix executables cannot be found')

    params= params or parameters_default()

    tmp_dir= work_dir or tempfile.mkdtemp()

    file_arguments, image_dim= _prepare_files([moving, fixed] + to_fit, tmp_dir)

    if not image_dim is None:
        params['FixedImageDimension']= image_dim
        params['MovingImageDimension']= image_dim
        if image_dim == 3 and params['ResultImageFormat'] is None:
            params['ResultImageFormat']= 'nii.gz'
        elif image_dim == 2:
            params['ResultImageFormat']= 'tif'
            params['ResultImagePixelType']= 'float'

    params_file= os.path.join(tmp_dir, 'params.txt')
    _save_parameters_to_file(params, params_file)
    
    command= [elastix_path,
              '-m',
              file_arguments[0],
              '-f',
              file_arguments[1],
              '-out',
              tmp_dir,
              '-p',
              params_file,
              '-threads',
              str(threads)]

    if verbose > 2: print('Executing the registration')
    
    _execute_command(command, verbose)
    
    if not registered_image_path is None:
        if image_dim == 3:
            if params['ResultImageFormat'] == "nii.gz":
                shutil.copyfile(os.path.join(tmp_dir, 'result.0.nii.gz'), registered_image_path)
            else:
                shutil.copyfile(os.path.join(tmp_dir, 'result.0.nii'), registered_image_path)
        elif image_dim == 2:
            shutil.copyfile(os.path.join(tmp_dir, 'result.0.tif'), registered_image_path)
    
    path_trafo_params = os.path.join(tmp_dir, 'TransformParameters.0.txt')

    results= []

    for i in range(len(file_arguments[2:])):
        if verbose > 2: print('Executing the fitting of %s' % file_arguments[2:][i])
        command= [transformix_path,
                  '-in',
                  file_arguments[2:][i],
                  '-out',
                  tmp_dir,
                  '-tp',
                  path_trafo_params,
                  '-threads',
                  str(threads)]
        
        _execute_command(command, verbose)
        
        if output_names is None:
            if image_dim == 3:
                if params['ResultImageFormat'] == "nii.gz":
                    results.append(nib.load(os.path.join(tmp_dir, 'result.nii.gz')))
                else:
                    results.append(nib.load(os.path.join(tmp_dir, 'result.nii')))
                results[-1].get_fdata()
            elif image_dim == 2:
                results.append(imageio.imread(os.path.join(tmp_dir, 'result.tif')))
        else:
            if image_dim == 3:
                if params['ResultImageFormat'] == "nii.gz":
                    shutil.copyfile(os.path.join(tmp_dir, 'result.nii.gz'), output_names[i])
                else:
                    shutil.copyfile(os.path.join(tmp_dir, 'result.nii'), output_names[i])
                results.append(output_names[i])
            elif image_dim == 2:
                shutil.copyfile(os.path.join(tmp_dir, 'result.tif'), output_names[i])
                results.append(output_names[i])
        
    if work_dir is None:
        shutil.rmtree(tmp_dir)
        
    return results

def compute_features(mask, image, threshold, bins, features, feature_names, features_to_compute, postfix):
    mask= mask > threshold
    masked_image= image[mask]
    
    if 'num' in features_to_compute:
        features.append(np.sum(mask))
        feature_names.append('-'.join([features_to_compute['num'], postfix]))
    if 'sum' in features_to_compute:
        features.append(np.sum(masked_image))
        feature_names.append('-'.join([features_to_compute['sum'], postfix]))
    if 'mean' in features_to_compute:
        features.append(np.mean(masked_image))
        feature_names.append('-'.join([features_to_compute['mean'], postfix]))
    if 'std' in features_to_compute:
        features.append(np.std(masked_image))
        feature_names.append('-'.join([features_to_compute['std'], postfix]))
    if 'skew' in features_to_compute:
        features.append(skew(masked_image))
        feature_names.append('-'.join([features_to_compute['skew'], postfix]))
    if 'kurt' in features_to_compute:
        features.append(kurtosis(masked_image))
        feature_names.append('-'.join([features_to_compute['kurt'], postfix]))
    if 'hist' in features_to_compute:
        features.extend(np.histogram(masked_image, bins=bins)[0])
        feature_names.extend(['-'.join(['hist', '%d' % k, postfix]) for k in range(len(bins)-1)])
    
    return features, feature_names

def extract_features_3d(image, 
                        masks,
                        labels,
                        thresholds= [0.5],
                        bins= [i*100 for i in range(0,12)],
                        features_to_compute={'num': 'num',
                                                'sum': 'sum',
                                                'mean': 'mean',
                                                'std': 'std',
                                                'skew': 'skew',
                                                'kurt': 'kurt',
                                                'hist': 'hist'}):
    """
    extract statistical descriptors
    Args:
        image (str, ndarray, Nifti1Image): original image
        masks (list(str, ndarray, Nifti1Image)): list of masks
        labels (list(str)): the list of labels to use in the feature names,
                            generally, the ids of the masks
        thresholds (list(float)): the thresholds used to threshold the masks
        bins (list(float)): list of bin boundaries used to compute histograms
    Returns:
        DataFrame: a dataframe of 1 line containing the extracted features
    """
    
    if isinstance(image, str):
        image= nib.load(image).get_fdata()
    elif isinstance(image, nib.Nifti1Image):
        image= image.get_fdata()
    
    features= []
    feature_names= []
    
    features, feature_names= [], []
    
    for m, i in zip(masks,range(len(masks))):
        for t in thresholds:
            if isinstance(m, str):
                mask= nib.load(m).get_fdata()
            elif isinstance(m, nib.Nifti1Image):
                mask= m.get_fdata()
            else:
                mask= m
            
            features, feature_names= compute_features(mask, 
                                                    image, 
                                                    t, 
                                                    bins, 
                                                    features, 
                                                    feature_names, 
                                                    features_to_compute, 
                                                    '%s-%f' % (labels[i], t))
    mean_mask_image= None
    for m, i in zip(masks, range(len(masks))):
        if isinstance(m, str):
            mask= nib.load(m).get_fdata()
        elif isinstance(m, nib.Nifti1Image):
            mask= m.get_fdata()
        else:
            mask= m
        
        if mean_mask_image is None:
            mean_mask_image= mask
        else:
            mean_mask_image+= mask
    mean_mask_image= mean_mask_image/len(masks)
    for t in thresholds:
        features, feature_names= compute_features(mean_mask_image, 
                                                    image, 
                                                    t, 
                                                    bins, 
                                                    features, 
                                                    feature_names, 
                                                    features_to_compute, 
                                                    '%f-mean_mask' % (t))
    
    return pd.DataFrame(data=[features], columns= feature_names)

def model_selection(features, 
            target, 
            objectives=[KNNR_Objective, 
                        LinearRegression_Objective, 
                        LassoRegression_Objective,
                        RidgeRegression_Objective,
                        PLSRegression_Objective],
            dataset=None,
            type=None,
            disable_feature_selection=False):
    all_results= []

    for o in objectives:
        print("Objective {}:".format(o.__name__))
        results={}
        ms= ModelSelection(o, 
                            features.values, 
                            target.values, 
                            verbosity=0, 
                            score_functions=[NegR2_score()], 
                            preprocessor=StandardScaler(), 
                            optimizer=SimulatedAnnealing(verbosity=0,
                                                            random_state=11),
                            random_state=11,
                            disable_feature_selection=disable_feature_selection)
        results['model_selection_score']= ms.select()['score']
        results['features']= list(features.columns[ms.get_best_model()["features"]])
        results['parameters']= ms.get_best_model()['model'].regressor.get_params()
        results['model']= o.__name__

        best = ms.get_best_model()
        used_features=[features.columns[i] for i, x in enumerate(best["features"]) if x]
        
        print("Number of used features: {}\nUsed features: {} \nScore: {}".format(len(used_features), used_features, best["score"]))
        for i in [1]:
            tmp= ms.evaluate(n_estimators=i, score_functions=[R2_score(), RMSE_score()], validator= RepeatedKFold(n_splits=10, n_repeats=20, random_state=21))
            results['r2_' + str(i)]= tmp['scores'][0]
            results['rmse_' + str(i)]= tmp['scores'][1]
            results['y_test_' + str(i)]= tmp['y_test']
            results['y_pred_' + str(i)]= tmp['y_pred']
            results['y_indices_' + str(i)]= tmp['y_indices']
            results['r2_per_fold_' + str(i)]= tmp['scores_per_fold'][0]
            results['rmse_per_fold_' + str(i)]= tmp['scores_per_fold'][1]
            results['r2_std_' + str(i)]= np.std(tmp['scores_per_fold'][0])
            results['rmse_std_' + str(i)]= np.std(tmp['scores_per_fold'][1])
            print(i, results['r2_' + str(i)])
        results['dataset']= dataset
        results['type']= type
        
        all_results.append(results)
    
    return pd.DataFrame(all_results)

# locating the elastix library
_maweight_executables= {'elastix': _find_executable('elastix'),
                      'transformix': _find_executable('transformix')}

if _maweight_executables['elastix'] == None:
    print('elastix executable not found, please pass its path to register_and_transform function')
if _maweight_executables['transformix'] == None:
    print('transformix executable not found, please pass its path to register_and_transform function')
if _maweight_executables['elastix'] and _maweight_executables['transformix']:
    print('Executables being used: %s %s' % (_maweight_executables['elastix'], _maweight_executables['transformix']))

