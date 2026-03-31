import os
from pathlib import Path
import nibabel as nib
from xml.etree import ElementTree
from sklearn.utils import Bunch


def get_aal(atlas_dir: str = "./external") -> Bunch:
    """
    Load AAL3 atlas from local files.
    Expected files in directory:
    - AAL3v1.nii or similar (.nii.gz also supported)
    - AAL3v1.xml or similar
    
    Args:
        atlas_dir: Directory containing AAL3 files

    Returns:
        Bunch object containing:
            - maps: path to the nifti file
            - labels: list of region names
            - indices: list of region indices
            - description: atlas description
    """
    atlas_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", atlas_dir)
    atlas_dir = Path(atlas_dir)
    
    # Find nifti file
    nifti_files = list(atlas_dir.glob("AAL3*.nii*"))
    if not nifti_files:
        raise FileNotFoundError(
            f"No AAL3 nifti file found in {atlas_dir}. "
            "Expected filename starting with 'AAL3' and ending with .nii or .nii.gz"
        )
    atlas_file = str(nifti_files[0])
    
    # Find XML file
    xml_files = list(atlas_dir.glob("AAL3*_1mm.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No AAL3 XML file found in {atlas_dir}. "
            "Expected filename starting with 'AAL3' and ending with .xml"
        )
    label_file = xml_files[0]
    
    # Read labels and indices from XML
    xml_tree = ElementTree.parse(label_file)
    root = xml_tree.getroot()
    
    labels = []
    indices = []
    
    for label in root.iter("label"):
        index_elem = label.find("index")
        name_elem = label.find("name")
        
        if index_elem is not None and name_elem is not None:
            indices.append(index_elem.text)
            labels.append(name_elem.text)
    
    # Create description
    description = """
    AAL3 atlas: Automated Anatomical Labeling version 3
    
    This atlas is the result of an automated anatomical parcellation of
    the spatially normalized single-subject high-resolution T1 volume
    provided by the Montreal Neurological Institute (MNI).
    
    The current version contains {n_labels} distinct regions.
    
    Note: The integers in the map image that define the parcellation
    are not consecutive and should not be interpreted as indices into
    the list of label names. Use the 'indices' list to map between
    label names and region values in the image.
    
    For more information, see:
    Rolls, E. T., Huang, C. C., Lin, C. P., Feng, J., & Joliot, M. (2020).
    Automated anatomical labelling atlas 3. Neuroimage, 206, 116189.
    """.format(n_labels=len(labels))
    
    return Bunch(
        maps=atlas_file,
        labels=labels,
        indices=indices,
        description=description
    )
