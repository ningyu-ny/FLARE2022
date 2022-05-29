import numpy as np
import SimpleITK as sitk
import torch

def float_uniform(low, high, size=None):
    """
    Create random floats in the lower and upper bounds - uniform distribution.
    :param low: Minimum value.
    :param high: Maximum value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    return float(values)
def int_uniform(low, high, size=None):
    """
    Create random integers in the lower and upper bounds (uniform distribution).
    :param low: Minimum value.
    :param high: Maximum value.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.uniform(low=float(low), high=float(high), size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.uint64)
    return int(values)
def bool_bernoulli(p=0.5, size=None):
    """
    Create random booleans with a given probability.
    :param p: Probabilities for the binomial distribution.
    :param size: If None, function returns a single value, otherwise random values in size of shape.
    :return: The random values.
    """
    values = np.random.binomial(n=1, p=p, size=size)
    if isinstance(values, np.ndarray):
        return values.astype(np.bool)
    return bool(values)
def index_to_physical_point(index, origin, direction, spacing):
    """
    Returns a physical point for an image index and given image metadata.
    :param index: The index to transform.
    :param origin: The image origin.
    :param spacing: The image spacing.
    :param direction: The image direction.
    :return: The transformed point.
    """
    dim = len(index)
    physical_point = np.array(origin) + np.matmul(np.matmul(np.array(direction).reshape([dim, dim]), np.diag(spacing)), np.array(index))
    return physical_point.tolist()
def img_physical_center(image):
   
    size = image.GetSize()  
    origin = image.GetOrigin()    
    direction = image.GetDirection()
    spacing = image.GetSpacing()
   
    return index_to_physical_point((np.array(size)-1)*0.5, origin, direction, spacing)
"""
Affine spatial transformations(translation, scaling & rotation): rotation, scaling will be performed around image center.
"""
def random_translation(image,
                       offset_factor = (0.05, 0.05, 0.05), # zyx
                       ):
    # Translation transform
   
    size = image.GetSize()  
    spacing = image.GetSpacing()
   
    dim = image.GetDimension()
    t = sitk.AffineTransform(dim)
   
    # translation in mm along each axis    
    current_offset = [float_uniform(-offset_factor[i], offset_factor[i])*size[i]*spacing[i]
                      for i in range(len(offset_factor))]
    t.Translate(current_offset)
    return t
def random_scale(image,
                 scale_factor = (0.05, 0.05, 0.05), # zyx
                 ):
    # Scale transform
   
    dim = image.GetDimension()
    t = sitk.AffineTransform(dim)
   
    # the bigger the value, the smaller the image    
    current_scale = [1.0 + float_uniform(-scale_factor[i], scale_factor[i])
                     for i in range(len(scale_factor))]
    t.Scale(current_scale)
    # t.SetCenter(img_physical_center(image))
    return t
def random_flip(image,
                flip_axes_probs = (0.5, 0, 0), # zyx
                ):
    # Flip transform
   
    dim = image.GetDimension()
    t = sitk.AffineTransform(dim)
   
    current_flip_axes = [bool(bool_bernoulli(p=flip_axes_probs[i]))
                         for i in range(dim)]
    # a flip is implemented by scaling the image axis by -1.0    
    current_scale = [-1.0 if f else 1.0 for f in current_flip_axes]
    t.Scale(current_scale)
    # t.SetCenter(img_physical_center(image))
    return t
def random_rotation(image,
                    random_angles = (5*np.pi/180, 5*np.pi/180, 5*np.pi/180), # zyx (in radians)
                    ):
    # Rotation transform
   
    dim = image.GetDimension()
    t = sitk.AffineTransform(dim)
   
    current_angles = [float_uniform(-random_angles[i], random_angles[i])
                      for i in range(dim)]
    # rotate about x axis
    t.Rotate(1, 2, angle=current_angles[2])
    # rotate about y axis
    t.Rotate(0, 2, angle=current_angles[1])
    # rotate about z axis
    t.Rotate(0, 1, angle=current_angles[0])
    # t.SetCenter(img_physical_center(image))
    return t
"""
The deformation spatial transformation randomly transforms points on an image grid and interpolates with splines.
"""    
def parse_num_control_points(
    num_control_points,
    ) -> None:
    for axis, number in enumerate(num_control_points):
        if not isinstance(number, int) or number < 4:
            message = (
                f'The number of control points for axis {axis} must be'
                f' an integer greater than 3, not {number}'
            )
            raise ValueError(message)
def parse_max_displacement(
        max_displacement,
        ) -> None:
    for axis, number in enumerate(max_displacement):
        if number < 0:
            message = (
                'The maximum displacement at each control point'
                f' for axis {axis} must be'
                f' a number greater or equal to 0'
            )
            raise ValueError(message)
           
def parse_free_form_transform(
        transform,
        max_displacement,
        ) -> None:
    """Issue a warning is possible folding is detected."""
    coefficient_images = transform.GetCoefficientImages()
    grid_spacing = coefficient_images[0].GetSpacing()
    conflicts = np.array(max_displacement) > np.array(grid_spacing) / 2
    if np.any(conflicts):
        where, = np.where(conflicts)
        message = (
            'The maximum displacement is larger than the coarse grid'
            f' spacing for dimensions: {where.tolist()}, so folding may'
            ' occur. Choose fewer control points or a smaller'
            ' maximum displacement'
        )
        raise ValueError(message)
   
def get_params(
        num_control_points,  
        max_displacement,  
        num_locked_borders,
        ) -> np.ndarray:
    grid_shape = num_control_points
    num_dimensions = 3
    # coarse_field = torch.rand(*grid_shape, num_dimensions)  # [0, 1)
    # coarse_field -= 0.5  # [-0.5, 0.5)
    # coarse_field *= 2    # [-1, 1]
    coarse_field = float_uniform(-1, 1, grid_shape+(num_dimensions,)) # [-1,1)
    for dimension in range(3):
        # [-max_displacement, max_displacement)
        coarse_field[..., dimension] *= max_displacement[dimension]
   
    # Set displacement to 0 at the borders
    for i in range(num_locked_borders):
        coarse_field[i, :] = 0
        coarse_field[-1 - i, :] = 0
        coarse_field[:, i] = 0
        coarse_field[:, -1 - i] = 0
   
    return coarse_field
 
def get_bspline_transform(
        image: sitk.Image,
        control_points: np.ndarray,
        ) -> sitk.BSplineTransformInitializer:
    SPLINE_ORDER = 3
    num_control_points = control_points.shape[:-1]
    mesh_shape = [n - SPLINE_ORDER for n in num_control_points]
    bspline_transform = sitk.BSplineTransformInitializer(image, mesh_shape)
    parameters = control_points.flatten(order='F').tolist()
    bspline_transform.SetParameters(parameters)
    return bspline_transform
# def get_bspline_transform(
#         image: sitk.Image,
#         control_points: np.ndarray,
#         spline_order: int = 3
#         ) -> sitk.BSplineTransform:
#     """
#     Returns the sitk transform based on the given image & parameters.
#     """
#     num_control_points = control_points.shape[:-1]
#     mesh_size = [n - spline_order for n in num_control_points]
   
#     deform_params = control_points.flatten(order='F').tolist()
   
#     dim = image.GetDimension()
#     input_size = image.GetSize()  
#     origin = image.GetOrigin()    
#     direction = image.GetDirection()
#     input_spacing = image.GetSpacing()
       
#     physical_dimensions = [input_size[i] * input_spacing[i] for i in range(dim)]
#     t = sitk.BSplineTransform(dim, spline_order)
#     t.SetTransformDomainOrigin(origin or np.zeros(dim))
#     t.SetTransformDomainMeshSize(mesh_size)
#     t.SetTransformDomainPhysicalDimensions(physical_dimensions)
#     t.SetTransformDomainDirection(direction or np.eye(dim).flatten())
#     t.SetParameters(deform_params)
#     return t
def random_deform(image,
                  num_control_points = (7, 7, 7),      
                  max_displacement = (7.5, 7.5, 7.5),
                  num_locked_borders = 0,
                  ):
    # BSplineDeformable transform
   
    # num_control_points:
    # Smaller numbers generate smoother deformations. The minimum number is
    # 4 as this transform uses cubic B-splines to interpolate displacement.
   
    # num_locked_borders:
    # If 0, all displacement vectors are kept.
    # If 1, displacement of control points at the
    # border of the coarse grid will be set to 0.
    # If 2, displacement of control points at the border of the image
    # (red dots in the image below) will also be set to 0.
   
    parse_num_control_points(num_control_points)
    parse_max_displacement(max_displacement)
     
    t = get_bspline_transform(image, get_params(num_control_points, max_displacement, num_locked_borders))
   
    parse_free_form_transform(t, max_displacement)
       
    return t
"""
A composite transformation consisting of multiple other consecutive transformations.
"""
def get_affine_homogeneous_matrix(dim, transformation):
    """
    Returns a homogeneous matrix for an affine transformation.
    :param dim: The dimension of the transformation.
    :param transformation: The sitk transformation.
    :return: A homogeneous (dim+1)x(dim+1) matrix as an np.array.
    """
    matrix = np.eye(dim + 1)
    matrix[:dim, :dim] = np.array(transformation.GetMatrix()).reshape([dim, dim]).T
    matrix[dim, :dim] = np.array(transformation.GetTranslation())
    return matrix
def get_affine_matrix_and_translation(dim, homogeneous_matrix):
    """
    Returns an affine transformation parameters for a homogeneous matrix.
    :param dim: The dimension of the transformation.
    :param homogeneous_matrix: The homogeneous (dim+1)x(dim+1) matrix as an np.array.
    :return: A tuple of the homogeneous matrix as a list, and the translation parameters as a list.
    """
    matrix = homogeneous_matrix[:dim, :dim].T.reshape(-1).tolist()
    translation = homogeneous_matrix[dim, :dim].tolist()
    return matrix, translation
def composite_transform(image,                
                        probs, # translate, scale, flip, rotate, deform
                        ):
    """
    Creates a composite sitk transform based on a list of sitk transforms.
    :param dim: The dimension of the transformation.
    :param transformations: A list of sitk transforms.
    :param merge_affine: If true, merge affine transformations before calculating the composite transformation.
    :return: The composite sitk transform.
    """  
   
    dim = image.GetDimension()  
   
    transformations = []
    # translate    
    if probs[0] > torch.rand(1): # [0,1)
        transformations.append(random_translation(image))
    # scale    
    if probs[1] > torch.rand(1):
        transformations.append(random_scale(image))
    # flip    
    if probs[2] > torch.rand(1):
        transformations.append(random_flip(image))
    # rotate    
    if probs[3] > torch.rand(1):
        transformations.append(random_rotation(image))
    # merge affine transform
    combined_matrix = None
    for transformation in transformations:
        if combined_matrix is None:
            combined_matrix = np.eye(dim + 1)
        current_matrix = get_affine_homogeneous_matrix(dim, transformation)
        combined_matrix = current_matrix @ combined_matrix
    if combined_matrix is not None:
        matrix, translation = get_affine_matrix_and_translation(dim, combined_matrix)
        combined_affine_transform = sitk.AffineTransform(dim)
        combined_affine_transform.SetMatrix(matrix)
        combined_affine_transform.SetTranslation(translation)
        combined_affine_transform.SetCenter(img_physical_center(image))
        transformations = [combined_affine_transform]
   
    # deform    
    if probs[4] > torch.rand(1):
        transformations.append(random_deform(image))        
             
    if len(transformations)==0:
        return transformations  
    if sitk.Version_MajorVersion() == 1:
        compos = sitk.Transform(dim, sitk.sitkIdentity)
        for transformation in transformations:
            compos.AddTransform(transformation)
    else:
        compos = sitk.CompositeTransform(transformations)
    return compos
def apply_transform(image,
                    transform,
                    defaultValue = float(0),
                    interpolators = sitk.sitkLinear):
   
    if transform==[]:
        return image
 
    outputSize = image.GetSize()  
    outputOrigin = image.GetOrigin()    
    outputDirection = image.GetDirection()
    outputSpacing = image.GetSpacing()
    interpolator = interpolators
    defaultPixelValue = defaultValue
    outputPixelType = image.GetPixelID()
   
    return sitk.Resample(image, outputSize, transform, interpolator,  
                         outputOrigin, outputSpacing, outputDirection,
                         defaultPixelValue, outputPixelType)

# convert numpy array to simpleitk img          
def sitkimg_from_nparr(self, arr, idx):            
    img = sitk.GetImageFromArray(arr.transpose((2,1,0))) # zyx->xyz
    img.SetOrigin(tuple(map(float,self.origins_A[idx])))
    img.SetDirection(tuple(map(float,self.directions_A[idx])))
    img.SetSpacing(tuple(map(float,self.spacings_A[idx])))
    return img

# data augmentation
def sitk_aug(img,mask):
    # if img_arr.ndim == 3:
        # img = self.sitkimg_from_nparr(img_arr, idx) # zyx->xyz->zyx 
    # img = sitk.ReadImage(pth_img)
    # mask = sitk.ReadImage(pth_mask)
    
    # aug
    t = composite_transform(img,
                    probs = (0.5, 0.5, 0.0, 0.5, 0.0))
                    #仿射，裁剪，对称，旋转，
    
    img = apply_transform(img, t)
    mask = apply_transform(mask, t)
    
    return img,mask#sitk.GetArrayFromImage(img).transpose((2,1,0)) # zyx->xyz->zyx
    # elif img_arr.ndim == 4:
    #     # 0-BG
    #     img_arr_sub = img_arr[0,:] # zyx
    #     img = self.sitkimg_from_nparr(img_arr_sub, idx) # zyx->xyz->zyx             
    #     # aug
    #     t = composite_transform(img,
    #                     probs = (0.0, 1.0, 1.0, 0.0, 0.0)) # translate, scale, flip, rotate, deform
    #     img = apply_transform(img, t, 1, sitk.sitkNearestNeighbor)            
    #     img_arr[0,:] = sitk.GetArrayFromImage(img).transpose((2,1,0)) # zyx->xyz->zyx
        
    #     for i in range(1, self.classes_nums):
    #         img_arr_sub = img_arr[i,:] # zyx
    #         img = self.sitkimg_from_nparr(img_arr_sub, idx) # zyx->xyz->zyx 
            
    #         img = apply_transform(img, t, 0, sitk.sitkNearestNeighbor)
            
    #         img_arr[i,:] = sitk.GetArrayFromImage(img).transpose((2,1,0)) # zyx->xyz->zyx
            
        # return img_arr

if __name__ == '__main__':
    img,mask = sitk_aug(sitk.ReadImage("input.nii.gz"),sitk.ReadImage("mask.nii.gz"))
    sitk.WriteImage(img,"outputimg.nii.gz")
    sitk.WriteImage(mask,"outputmask.nii.gz")